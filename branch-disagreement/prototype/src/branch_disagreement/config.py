"""Experiment configuration."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class ExperimentConfig:
    """All knobs for one experiment run.

    The same config object drives the proxy (CPU) and vLLM (GPU) paths; only
    ``engine`` decides which runner is constructed.
    """

    # engine: "proxy" (CPU, synthetic) or "vllm" (GPU, real model)
    engine: str = "proxy"

    # model / dataset
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    nli_model: str = "microsoft/deberta-large-mnli"
    dataset: str = "sample"  # "sample" | "popqa" | "triviaqa"
    split: str = ""  # "" = per-dataset default (popqa->test, triviaqa->validation)
    limit: int = 12

    # "short": terse one-line factual answer (exact-match-friendly).
    # "long":  free-form sentence(s) — the regime where semantic clustering and
    #          token-logprob diverge (the semantic-entropy use case).
    response_mode: str = "short"

    # branch sampling
    n_branches: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 64
    # "self_consistency": N samples of the same prompt.
    # "localised": N branches, each with a distinct approach marker after the
    #              shared prefix (the localised-reasoning steering protocol).
    branch_mode: str = "self_consistency"

    # semantic entropy weighting: "count" (discrete) or "likelihood".
    entropy_weighting: str = "count"

    # bootstrap / determinism
    bootstrap_samples: int = 1000
    seed: int = 0

    # output
    output_prefix: str = "branch_disagreement"
    reports_dir: str = "reports"

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
