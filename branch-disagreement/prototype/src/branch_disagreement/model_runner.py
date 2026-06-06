"""Model runners that produce prefix-shared branches for a question.

A "branch" is one sampled continuation from a shared prefix. Two runners share
one interface (``generate_branches``):

- ``ProxyRunner``: CPU-only, deterministic, synthetic. Uses the sample dataset's
  ``p_known`` hint to decide whether the model "knows" the fact, then produces
  branch answers that either agree on the gold (known) or scatter across
  distractors (unknown). It also emits plausible token logprobs so the
  confidence baseline is computable. A process check, never a quality claim.
- ``VLLMRunner``: loads a real model under vLLM and samples N branches. vLLM's
  automatic prefix caching makes the shared prefix near-free across branches.
  Records generated-token counts for the cost axis.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .datasets import EvalQuestion

# Distinct approach markers for the localised branch mode. Each marker is a small
# steering suffix after the shared prefix; together they probe whether different
# framings of the same question converge (model knows) or diverge (it does not).
# Truth-preserving perspective markers: each pushes a genuinely different
# reasoning route to the SAME fact, rather than a stylistic tweak. The intent is
# orthogonal divergence in approach with convergence on truth, so a fact the
# model knows survives all routes (low disagreement) while a guess scatters.
LOCALISED_MARKERS = [
    "Recall the specific fact directly and state it.",
    "Recall it via the most authoritative source you know, then answer.",
    "Cross-check against a second independent fact about the subject, then answer.",
    "Rule out the options you are confident are wrong, then answer.",
    "Reason from closely related facts you are certain of, then answer.",
    "Fix the relevant time period and context first, then answer.",
    "Avoid your first guess; verify it against what you know, then answer.",
    "State what you are certain of about the subject, then give the answer it implies.",
]


@dataclass
class BranchSample:
    branch_id: int
    text: str                       # full generated continuation
    answer: str                     # extracted short answer
    token_logprobs: List[float] = field(default_factory=list)
    num_tokens: int = 0
    marker: Optional[str] = None

    @property
    def mean_logprob(self) -> float:
        if not self.token_logprobs:
            return 0.0
        return sum(self.token_logprobs) / len(self.token_logprobs)


@dataclass
class GenerationResult:
    branches: List[BranchSample]
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_s: float = 0.0


def build_shared_prefix(question: str, system: Optional[str] = None) -> str:
    system = system or (
        "You are a precise assistant. Answer the question with the single most "
        "likely short factual answer. If you are not sure, still give your best "
        "single answer."
    )
    return f"{system}\n\nQuestion: {question}\nAnswer:"


def markers_for(branch_mode: str, n_branches: int) -> List[Optional[str]]:
    if branch_mode == "localised":
        return [LOCALISED_MARKERS[i % len(LOCALISED_MARKERS)] for i in range(n_branches)]
    return [None] * n_branches


class ProxyRunner:
    """Deterministic synthetic runner for CPU smoke tests."""

    engine = "proxy"

    def __init__(self, seed: int = 0):
        self.base_seed = seed

    def name(self) -> str:
        return "proxy"

    def generate_branches(
        self,
        question: EvalQuestion,
        n_branches: int,
        temperature: float = 0.8,
        branch_mode: str = "self_consistency",
        max_new_tokens: int = 64,
        response_mode: str = "short",  # accepted for interface parity; unused
    ) -> GenerationResult:
        rng = random.Random(f"{self.base_seed}:{question.id}")
        p_known = float(question.metadata.get("p_known", 0.5))
        gold = question.gold_answers[0] if question.gold_answers else "unknown"
        distractors = list(question.metadata.get("distractors", [])) or ["option a", "option b"]
        markers = markers_for(branch_mode, n_branches)

        # One latent "does the model know this" draw per question, then branch
        # answers are conditionally sampled. Known -> concentrate on gold;
        # unknown -> scatter across gold + distractors with gold not dominant.
        knows = rng.random() < p_known
        branches: List[BranchSample] = []
        gen_tokens = 0
        for b in range(n_branches):
            if knows:
                ans = gold if rng.random() < 0.9 else rng.choice(distractors)
                base_lp = -0.25
            else:
                pool = [gold] + distractors
                # gold is only weakly preferred when unknown
                weights = [0.30] + [0.70 / len(distractors)] * len(distractors)
                ans = rng.choices(pool, weights=weights, k=1)[0]
                base_lp = -1.4
            ntok = max(1, len(ans.split()) + rng.randint(0, 3))
            gen_tokens += ntok
            logprobs = [base_lp + rng.uniform(-0.3, 0.3) for _ in range(ntok)]
            branches.append(
                BranchSample(
                    branch_id=b,
                    text=ans,
                    answer=ans,
                    token_logprobs=logprobs,
                    num_tokens=ntok,
                    marker=markers[b],
                )
            )
        return GenerationResult(
            branches=branches,
            prompt_tokens=len(build_shared_prefix(question.question).split()),
            generated_tokens=gen_tokens,
            latency_s=0.0,
        )


class VLLMRunner:  # pragma: no cover - GPU-box path
    """Real model runner backed by vLLM with shared-prefix branch sampling."""

    engine = "vllm"

    def __init__(
        self,
        model: str,
        max_model_len: int = 4096,
        # 0.80 (not 0.90) leaves ~2.4GB on a 24GB card for the NLI clusterer
        # model, which loads onto the same GPU after vLLM. At 0.90 the NLI model
        # OOMs and clustering silently degrades to exact match.
        gpu_memory_utilization: float = 0.80,
        seed: int = 0,
        dtype: str = "auto",
    ):
        import os

        # vLLM 0.22 defaults to a FlashInfer sampler that JIT-compiles a CUDA
        # kernel at startup, which needs CUDA dev headers (curand.h) absent from
        # bare images. Force the native Torch sampler — no runtime compilation.
        # Must be set before importing vllm.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from vllm import LLM

        self.model = model
        self.seed = seed
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enable_prefix_caching=True,  # the shared-prefix mechanism
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def name(self) -> str:
        return self.model

    _SYSTEM_SHORT = (
        "You are a precise factual assistant. Answer with ONLY the short factual "
        "answer (a name, term, or number) — no explanation, no full sentence. "
        "If unsure, give your single best guess."
    )
    _SYSTEM_LONG = (
        "You are a knowledgeable assistant. Answer the question in one or two "
        "complete sentences. Briefly explain, then clearly state the answer. "
        "If unsure, still commit to your single best answer."
    )

    def _build_prompt(self, question: str, marker: Optional[str], mode: str) -> str:
        """Build the chat prompt directly in Qwen's ChatML format.

        We deliberately do NOT use tokenizer.apply_chat_template: vLLM's
        get_tokenizer() can return the slow Qwen2Tokenizer, which trips
        transformers' template machinery (missing all_special_tokens_extended).
        Hand-building ChatML is version-proof and prefix-caches identically.
        The system + question form the shared prefix; a localised marker, if any,
        is appended to the user turn.
        """
        system = self._SYSTEM_LONG if mode == "long" else self._SYSTEM_SHORT
        user = question if marker is None else f"{question}\n(Approach: {marker})"
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _extract_answer(self, text: str, mode: str) -> str:
        text = text.strip()
        if mode == "long":
            # Keep the full free-form generation (collapsed whitespace) so NLI
            # clustering and gold-containment scoring operate on full meaning.
            return " ".join(text.split())
        line = text.splitlines()[0] if text else ""
        return line.strip().strip(".").strip().strip('"').strip()

    def generate_branches(
        self,
        question: EvalQuestion,
        n_branches: int,
        temperature: float = 0.8,
        branch_mode: str = "self_consistency",
        max_new_tokens: int = 64,
        response_mode: str = "short",
    ) -> GenerationResult:
        import time

        from vllm import SamplingParams

        markers = markers_for(branch_mode, n_branches)
        prompts = [
            self._build_prompt(question.question, m, response_mode) for m in markers
        ]

        # Short mode stops at the first newline (one-line answer). Long mode lets
        # the model write full sentences; only a hard double-newline stops it.
        stop = ["\n\n"] if response_mode == "long" else ["\n"]

        # In self-consistency mode every prompt is identical, so prefix caching
        # serves all branches from one prefill. Each branch gets a DISTINCT seed
        # so the samples actually diverge — a shared seed would make vLLM return
        # identical deterministic completions and kill the disagreement signal.
        params = [
            SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_new_tokens,
                logprobs=1,
                seed=self.seed + b,
                n=1,
                stop=stop,
            )
            for b in range(n_branches)
        ]
        t0 = time.time()
        outputs = self.llm.generate(prompts, params)
        latency = time.time() - t0

        branches: List[BranchSample] = []
        gen_tokens = 0
        prompt_tokens = 0
        for b, out in enumerate(outputs):
            comp = out.outputs[0]
            text = comp.text
            token_logprobs = self._token_logprobs(comp)
            ntok = len(comp.token_ids)
            gen_tokens += ntok
            prompt_tokens = len(out.prompt_token_ids)
            branches.append(
                BranchSample(
                    branch_id=b,
                    text=text,
                    answer=self._extract_answer(text, response_mode),
                    token_logprobs=token_logprobs,
                    num_tokens=ntok,
                    marker=markers[b],
                )
            )
        return GenerationResult(
            branches=branches,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            latency_s=latency,
        )

    @staticmethod
    def _token_logprobs(comp) -> List[float]:
        out = []
        for lp in comp.logprobs or []:
            if not lp:
                continue
            # vLLM logprobs: {token_id: Logprob(logprob=...)}; take the max.
            best = max(lp.values(), key=lambda x: x.logprob)
            out.append(best.logprob)
        return out


def build_runner(config) -> "ProxyRunner | VLLMRunner":
    if config.engine == "proxy":
        return ProxyRunner(seed=config.seed)
    if config.engine == "vllm":  # pragma: no cover - GPU-box path
        return VLLMRunner(model=config.model, seed=config.seed)
    raise ValueError(f"unknown engine: {config.engine!r}")
