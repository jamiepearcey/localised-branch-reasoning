#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from localised_reasoning.hf_kv_branching import (  # noqa: E402
    DEFAULT_HF_MODEL,
    DEFAULT_FORBIDDEN_VISIBLE_CONTROL_PHRASES,
    load_causal_lm,
    run_real_kv_fork,
)
from localised_reasoning.planned_branching import (  # noqa: E402
    DEFAULT_CONSIDERATIONS,
    parse_branch_plan,
    render_planner_prompt,
    rewrite_stacked_factors,
    stacked_marker_for_report,
)


DEFAULT_TASK = """\
We need to advance a KV-cache branching runtime. The next decision is whether
to continue directly or fork localized continuations that separately consider
implementation work, runtime risk, and regression coverage.
"""


FORBIDDEN_PLANNED_DEMO_LEAKS = DEFAULT_FORBIDDEN_VISIBLE_CONTROL_PHRASES + (
    "decision_point",
    "factors:",
    "selected factors",
    "active factor",
    "factor directive",
    "branch marker",
    "write one",
    "followed by why",
    "this branch is complete",
    "runtime-risk sentence",
    "implementation sentence",
    "regression-test sentence",
    "build factor",
    "review factor",
    "test factor",
    "result:",
    "do not repeat these instructions",
    "short reason this branch is complete",
)

MODEL_STOP_PHRASE = "STOP_POINT:"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Let a real model decide a decision point and factors, rewrite the "
            "stacked factor plan into one hidden marker per branch, then run a "
            "real KV fork."
        )
    )
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--planner-max-new-tokens", type=int, default=48)
    parser.add_argument("--continuation-max-new-tokens", type=int, default=48)
    parser.add_argument("--stop-extra-tokens", type=int, default=12)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Optional transformers attention implementation, for example eager.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only models already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ROOT / "reports" / "planned_branch_kv_demo.txt",
    )
    args = parser.parse_args()

    device = None
    if args.device != "auto":
        device = torch.device(args.device)

    planner_prompt = render_planner_prompt(
        task=args.task,
        considerations=DEFAULT_CONSIDERATIONS,
    )
    raw_plan = _generate_planner_output(
        model_name=args.model,
        prompt=planner_prompt,
        max_new_tokens=args.planner_max_new_tokens,
        device=device,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )

    allowed_labels = tuple(consideration.label for consideration in DEFAULT_CONSIDERATIONS)
    plan = parse_branch_plan(raw_plan, allowed_labels=allowed_labels)
    branch_markers = rewrite_stacked_factors(
        plan,
        considerations=DEFAULT_CONSIDERATIONS,
    )

    # The planner output is a control signal. The branch prefix is rewritten so
    # the stacked factor row is not carried into every continuation.
    fork_prefix = _render_rewritten_fork_prefix(
        task=args.task,
        decision_point=plan.decision_point,
    )

    result = run_real_kv_fork(
        model_name=args.model,
        prefix=fork_prefix,
        branch_markers=branch_markers,
        max_new_tokens=args.continuation_max_new_tokens,
        device=device,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        forbidden_visible_phrases=FORBIDDEN_PLANNED_DEMO_LEAKS,
        model_stop_phrase=MODEL_STOP_PHRASE,
        stop_extra_tokens_after_phrase=args.stop_extra_tokens,
    )

    transcript = _format_report(
        planner_prompt=planner_prompt,
        plan_raw=raw_plan,
        plan_labels=plan.selected_labels,
        decision_point=plan.decision_point,
        used_default_factors=plan.used_default_factors,
        stacked_marker=stacked_marker_for_report(plan),
        branch_markers=branch_markers,
        result=result,
    )
    print(transcript, end="")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(transcript, encoding="utf-8")
    print(f"output_file={args.output_file}")

    if len(result.branches) != len(plan.selected_factors):
        raise SystemExit("number of real branches does not match parsed factors")
    if plan.used_default_factors:
        raise SystemExit("planner did not emit parseable factors; defaults were used")
    if "no branching" in plan.decision_point.lower():
        raise SystemExit("planner declined to create a decision point")
    if not _branch_markers_are_single_factor(result.branches):
        raise SystemExit("a rewritten branch marker still contains stacked factors")
    if any(branch.visible_control_leaks for branch in result.branches):
        leaking = ", ".join(
            f"{branch.name}: {list(branch.visible_control_leaks)}"
            for branch in result.branches
        )
        raise SystemExit(f"visible output leaked control-language phrases: {leaking}")
    unstopped = [branch.name for branch in result.branches if not branch.model_stop_detected]
    if unstopped:
        raise SystemExit(f"branches did not emit model stop points: {unstopped}")


@torch.inference_mode()
def _generate_planner_output(
    *,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    device: torch.device | None,
    local_files_only: bool,
    attn_implementation: str | None,
) -> str:
    tokenizer, model, device = load_causal_lm(
        model_name,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    else:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _format_report(
    *,
    planner_prompt: str,
    plan_raw: str,
    plan_labels: tuple[str, ...],
    decision_point: str,
    used_default_factors: bool,
    stacked_marker: str,
    branch_markers: dict[str, str],
    result,
) -> str:
    lines = [
        f"model={result.model}",
        f"device={result.device}",
        f"planner_selected_factors={list(plan_labels)}",
        f"planner_used_default_factors={used_default_factors}",
        f"parallel_continuation_count={len(plan_labels)}",
        "=== planner_prompt ===",
        planner_prompt.strip(),
        "=== raw_planner_output ===",
        plan_raw.strip(),
        "=== stacked_marker_before_rewrite ===",
        stacked_marker,
        "=== rewritten_shared_prefix ===",
        _render_rewritten_fork_prefix(
            task=_extract_task_from_prompt(planner_prompt),
            decision_point=decision_point,
        ).strip(),
        "=== rewritten_branch_markers ===",
    ]
    for name, marker in branch_markers.items():
        lines.append(f"## branch_marker={name}")
        lines.append(repr(marker))
    lines.extend(
        [
            "=== kv_fork_result ===",
            f"prefix_forward_calls={result.prefix_forward_calls}",
            f"prefix_token_count={result.prefix_token_count}",
            f"prefix_prefill_seconds={result.prefix_prefill_seconds:.3f}",
            f"fork_shared_prefix_storage={result.fork_shared_prefix_storage}",
            f"prefix_unchanged_after_branches={result.prefix_unchanged_after_branches}",
        ]
    )
    for branch in result.branches:
        lines.extend(
            [
                f"## branch={branch.name}",
                f"marker_token_count={branch.marker_token_count}",
                f"prefix_length_before={branch.prefix_length_before}",
                f"length_after_marker={branch.length_after_marker}",
                f"length_after_visible={branch.length_after_visible}",
                (
                    "fork_vs_full_recompute_logits_max_abs_diff="
                    f"{branch.fork_logits_max_abs_diff:.8f}"
                ),
                f"suffix_checksum={branch.suffix_checksum:.4f}",
                f"model_stop_detected={branch.model_stop_detected}",
                f"model_stop_point={branch.model_stop_point!r}",
                f"visible_control_leaks={list(branch.visible_control_leaks)}",
                f"visible_text={branch.visible_text!r}",
            ]
        )
    return "\n".join(lines) + "\n"


def _branch_markers_are_single_factor(branches) -> bool:
    factor_phrases = {
        "build": "implementation step",
        "review": "runtime failure mode",
        "test": "regression test",
    }
    all_phrases = tuple(factor_phrases.values())
    for branch in branches:
        expected = factor_phrases.get(branch.name)
        if expected is None or expected not in branch.marker:
            return False
        unexpected = [phrase for phrase in all_phrases if phrase != expected]
        if any(phrase in branch.marker for phrase in unexpected):
            return False
    return True


def _render_rewritten_fork_prefix(*, task: str, decision_point: str) -> str:
    return f"""\
Shared task:
{task.strip()}

Planner decision point:
{decision_point.strip()}

The runtime will now evaluate one factor per continuation.
Every continuation must produce useful branch content and then a visible
STOP_POINT line explaining why that branch is complete.
"""


def _extract_task_from_prompt(prompt: str) -> str:
    marker = "Task:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\nOutput exactly", 1)[0].strip()


if __name__ == "__main__":
    main()
