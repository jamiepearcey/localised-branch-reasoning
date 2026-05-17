from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import inspect
import re
import time
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_FORBIDDEN_VISIBLE_CONTROL_PHRASES = (
    "hidden branch control",
    "branch control marker",
    "implementation branch",
    "runtime reviewer branch",
    "answer as",
)


@dataclass(frozen=True)
class CacheTensorInfo:
    shape: tuple[int, ...]
    data_ptr: int
    checksum: float


@dataclass(frozen=True)
class BranchRun:
    name: str
    marker: str
    marker_token_count: int
    fork_logits_max_abs_diff: float
    prefix_length_before: int
    length_after_marker: int
    length_after_visible: int
    visible_token_ids: tuple[int, ...]
    visible_text: str
    visible_control_leaks: tuple[str, ...]
    suffix_checksum: float
    model_stop_detected: bool = False
    model_stop_point: str = ""


@dataclass(frozen=True)
class RealKVForkRun:
    model: str
    device: str
    prefix_token_count: int
    prefix_prefill_seconds: float
    prefix_forward_calls: int
    prefix_unchanged_after_branches: bool
    fork_shared_prefix_storage: bool
    branches: tuple[BranchRun, ...]


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_causal_lm(
    model_name: str,
    *,
    device: torch.device | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
):
    device = device or best_device()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, object] = {"local_files_only": local_files_only}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def shallow_fork_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    """Create a branch cache whose layers initially reference the same prefix tensors.

    DynamicCache appends by assigning `layer.keys = torch.cat(...)`, so updating
    the forked layer materializes that branch while leaving the original prefix
    cache object untouched. This avoids repeated prefix prefill, but it is not a
    paged-attention copy-on-write implementation.
    """

    fork = DynamicCache()
    fork.layers = [copy(layer) for layer in cache.layers]
    fork.layer_class_to_replicate = None
    fork.offloading = getattr(cache, "offloading", False)
    if hasattr(cache, "only_non_sliding"):
        fork.only_non_sliding = cache.only_non_sliding
    return fork


def cache_seq_length(cache: DynamicCache) -> int:
    return int(cache.get_seq_length())


def cache_tensor_infos(cache: DynamicCache) -> tuple[CacheTensorInfo, ...]:
    infos: list[CacheTensorInfo] = []
    for tensor in _cache_tensors(cache):
        infos.append(
            CacheTensorInfo(
                shape=tuple(tensor.shape),
                data_ptr=tensor.data_ptr(),
                checksum=_checksum(tensor),
            )
        )
    return tuple(infos)


def cache_tensors_equal(left: DynamicCache, right_snapshots: tuple[torch.Tensor, ...]) -> bool:
    for tensor, snapshot in zip(_cache_tensors(left), right_snapshots, strict=True):
        if not torch.equal(tensor.detach().cpu(), snapshot):
            return False
    return True


def clone_cache_tensors_to_cpu(cache: DynamicCache) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.detach().cpu().clone() for tensor in _cache_tensors(cache))


def caches_share_tensor_storage(left: DynamicCache, right: DynamicCache) -> bool:
    left_ptrs = [tensor.data_ptr() for tensor in _cache_tensors(left)]
    right_ptrs = [tensor.data_ptr() for tensor in _cache_tensors(right)]
    return left_ptrs == right_ptrs and bool(left_ptrs)


@torch.inference_mode()
def run_real_kv_fork(
    *,
    model_name: str = DEFAULT_HF_MODEL,
    prefix: str,
    branch_markers: dict[str, str],
    max_new_tokens: int,
    device: torch.device | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
    forbidden_visible_phrases: tuple[str, ...] = DEFAULT_FORBIDDEN_VISIBLE_CONTROL_PHRASES,
    model_stop_phrase: str | None = None,
    stop_extra_tokens_after_phrase: int = 12,
) -> RealKVForkRun:
    tokenizer, model, device = load_causal_lm(
        model_name,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    encoded_prefix = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    prefix_ids = encoded_prefix["input_ids"].to(device)
    prefix_attention_mask = torch.ones_like(prefix_ids, device=device)

    start = time.perf_counter()
    prefix_out = _forward_with_cache(
        model,
        input_ids=prefix_ids,
        attention_mask=prefix_attention_mask,
        cache_position=torch.arange(prefix_ids.shape[-1], device=device),
    )
    prefix_prefill_seconds = time.perf_counter() - start
    prefix_cache = prefix_out.past_key_values
    if not isinstance(prefix_cache, DynamicCache):
        prefix_cache = DynamicCache.from_legacy_cache(prefix_cache)

    prefix_snapshots = clone_cache_tensors_to_cpu(prefix_cache)
    prefix_length = cache_seq_length(prefix_cache)
    branch_runs: list[BranchRun] = []

    initial_forks = [shallow_fork_dynamic_cache(prefix_cache) for _ in branch_markers]
    fork_shared_prefix_storage = all(
        caches_share_tensor_storage(prefix_cache, fork) for fork in initial_forks
    )

    for (name, marker), branch_cache in zip(branch_markers.items(), initial_forks, strict=True):
        marker_ids = tokenizer(marker, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)
        if marker_ids.numel() == 0:
            raise ValueError(f"marker for branch {name!r} tokenized to zero tokens")

        marker_attention_mask = torch.ones(
            (1, prefix_length + marker_ids.shape[-1]),
            dtype=torch.long,
            device=device,
        )
        marker_cache_position = torch.arange(
            prefix_length,
            prefix_length + marker_ids.shape[-1],
            device=device,
        )
        fork_marker_out = _forward_with_cache(
            model,
            input_ids=marker_ids,
            past_key_values=branch_cache,
            attention_mask=marker_attention_mask,
            cache_position=marker_cache_position,
        )
        branch_cache_after_marker = fork_marker_out.past_key_values
        if not isinstance(branch_cache_after_marker, DynamicCache):
            branch_cache_after_marker = DynamicCache.from_legacy_cache(
                branch_cache_after_marker
            )
        length_after_marker = cache_seq_length(branch_cache_after_marker)

        full_ids = torch.cat([prefix_ids, marker_ids], dim=-1)
        full_marker_out = _forward_with_cache(
            model,
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids, device=device),
            cache_position=torch.arange(full_ids.shape[-1], device=device),
        )
        logits_diff = torch.max(
            torch.abs(
                fork_marker_out.logits[:, -1, :].detach().float().cpu()
                - full_marker_out.logits[:, -1, :].detach().float().cpu()
            )
        ).item()

        visible_generation = _greedy_generate_from_cache(
            model=model,
            tokenizer=tokenizer,
            cache=branch_cache_after_marker,
            logits=fork_marker_out.logits,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            model_accepts_cache_position=_model_accepts(model, "cache_position"),
            model_stop_phrase=model_stop_phrase,
            stop_extra_tokens_after_phrase=stop_extra_tokens_after_phrase,
        )
        visible_token_ids = visible_generation.token_ids
        visible_text = tokenizer.decode(
            list(visible_token_ids),
            skip_special_tokens=True,
        )
        model_stop_point = extract_model_stop_point(
            visible_text,
            model_stop_phrase,
        )
        visible_control_leaks = visible_control_leaks_for(
            visible_text,
            forbidden_visible_phrases,
        )
        branch_runs.append(
            BranchRun(
                name=name,
                marker=marker,
                marker_token_count=int(marker_ids.shape[-1]),
                fork_logits_max_abs_diff=float(logits_diff),
                prefix_length_before=prefix_length,
                length_after_marker=length_after_marker,
                length_after_visible=cache_seq_length(branch_cache_after_marker),
                visible_token_ids=tuple(int(token_id) for token_id in visible_token_ids),
                visible_text=visible_text,
                visible_control_leaks=visible_control_leaks,
                suffix_checksum=_suffix_checksum(
                    branch_cache_after_marker,
                    prefix_length,
                ),
                model_stop_detected=bool(model_stop_point),
                model_stop_point=model_stop_point,
            )
        )

    return RealKVForkRun(
        model=model_name,
        device=str(device),
        prefix_token_count=prefix_length,
        prefix_prefill_seconds=prefix_prefill_seconds,
        prefix_forward_calls=1,
        prefix_unchanged_after_branches=cache_tensors_equal(
            prefix_cache,
            prefix_snapshots,
        ),
        fork_shared_prefix_storage=fork_shared_prefix_storage,
        branches=tuple(branch_runs),
    )


def _cache_tensors(cache: DynamicCache) -> Iterable[torch.Tensor]:
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        yield layer.keys
        yield layer.values


def _checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().sum().cpu().item())


def _suffix_checksum(cache: DynamicCache, prefix_length: int) -> float:
    total = 0.0
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        total += _checksum(layer.keys[..., prefix_length:, :])
        total += _checksum(layer.values[..., prefix_length:, :])
    return total


def visible_control_leaks_for(
    visible_text: str,
    forbidden_phrases: tuple[str, ...] = DEFAULT_FORBIDDEN_VISIBLE_CONTROL_PHRASES,
) -> tuple[str, ...]:
    lowered = visible_text.lower()
    return tuple(phrase for phrase in forbidden_phrases if phrase in lowered)


def extract_model_stop_point(
    visible_text: str,
    model_stop_phrase: str | None,
) -> str:
    if not model_stop_phrase:
        return ""
    pattern = re.compile(
        rf"{re.escape(model_stop_phrase)}\s*(.+?)(?:\n|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(visible_text)
    if not match:
        return ""
    return " ".join(match.group(1).strip().split())


@dataclass(frozen=True)
class GeneratedTokens:
    token_ids: tuple[int, ...]


def _greedy_generate_from_cache(
    *,
    model,
    tokenizer,
    cache: DynamicCache,
    logits: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    model_accepts_cache_position: bool,
    model_stop_phrase: str | None = None,
    stop_extra_tokens_after_phrase: int = 12,
) -> GeneratedTokens:
    generated: list[int] = []
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    stop_phrase_seen_at: int | None = None
    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if eos_token_id is not None and token_id == eos_token_id:
            break
        generated.append(token_id)
        should_stop_after_cache_update = False
        if model_stop_phrase and stop_phrase_seen_at is None:
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            if model_stop_phrase.lower() in decoded.lower():
                stop_phrase_seen_at = len(generated)
        elif stop_phrase_seen_at is not None:
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            stop_point = extract_model_stop_point(decoded, model_stop_phrase)
            extra_tokens = len(generated) - stop_phrase_seen_at
            if (
                "\n" in decoded.lower().split(model_stop_phrase.lower(), 1)[-1]
                or (
                    stop_point.endswith((".", "!", "?"))
                    and extra_tokens >= 3
                )
                or extra_tokens >= stop_extra_tokens_after_phrase
            ):
                should_stop_after_cache_update = True
        cache_position = torch.tensor(
            [cache_seq_length(cache)],
            dtype=torch.long,
            device=next_token.device,
        )
        attention_mask = torch.ones(
            (1, cache_seq_length(cache) + 1),
            dtype=torch.long,
            device=next_token.device,
        )
        kwargs = {
            "input_ids": next_token,
            "past_key_values": cache,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        if model_accepts_cache_position:
            kwargs["cache_position"] = cache_position
        out = model(**kwargs)
        cache = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if should_stop_after_cache_update:
            break
    return GeneratedTokens(token_ids=tuple(generated))


def _forward_with_cache(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: DynamicCache | None = None,
    cache_position: torch.Tensor | None = None,
):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    if cache_position is not None and _model_accepts(model, "cache_position"):
        kwargs["cache_position"] = cache_position
    return model(**kwargs)


def _model_accepts(model, parameter: str) -> bool:
    return parameter in inspect.signature(model.forward).parameters
