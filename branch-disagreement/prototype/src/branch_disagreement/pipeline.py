"""End-to-end experiment: generate -> cluster -> score -> label -> metrics."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .clustering import ExactMatchClusterer
from .config import ExperimentConfig
from .datasets import EvalQuestion, load_dataset
from .metrics import bootstrap_auc_ci, delong_test
from .normalize import answer_matches
from .scoring import SCORE_NAMES, compute_scores, primary_answer


@dataclass
class QuestionRow:
    id: str
    question: str
    gold_answers: List[str]
    predicted: str
    is_correct: bool
    is_error: int               # 1 = wrong (positive class for detection)
    scores: Dict[str, float]
    generated_tokens: int
    latency_s: float
    branch_answers: List[str] = field(default_factory=list)
    branch_logprobs: List[float] = field(default_factory=list)  # per-branch mean logprob
    branch_clusters: List[int] = field(default_factory=list)     # per-branch cluster id
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorSummary:
    name: str
    auc: float
    ci_low: float
    ci_high: float
    # DeLong vs the branch_disagreement detector (None for itself)
    delong_z_vs_branch: Optional[float] = None
    delong_p_vs_branch: Optional[float] = None


@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    rows: List[QuestionRow]
    detectors: List[DetectorSummary]
    n_questions: int
    n_correct: int
    n_error: int
    accuracy: float
    total_generated_tokens: int
    mean_tokens_per_question: float
    total_latency_s: float
    notes: List[str] = field(default_factory=list)


def _cluster(clusterer, answers, context):
    # NLIClusterer.cluster accepts a context kwarg; ExactMatchClusterer does not.
    try:
        return clusterer.cluster(answers, context=context)
    except TypeError:
        return clusterer.cluster(answers)


def run_experiment(
    config: ExperimentConfig,
    runner,
    clusterer=None,
    questions: Optional[List[EvalQuestion]] = None,
) -> ExperimentResult:
    if clusterer is None:
        clusterer = ExactMatchClusterer()
    if questions is None:
        questions = load_dataset(
            config.dataset, limit=config.limit, split=config.split or None
        )

    rows: List[QuestionRow] = []
    notes: List[str] = []
    if getattr(clusterer, "name", "") == "exact_match" and config.engine == "vllm":
        notes.append("NLI clusterer unavailable; used exact-match fallback.")

    for q in questions:
        gen = runner.generate_branches(
            q,
            n_branches=config.n_branches,
            temperature=config.temperature,
            branch_mode=config.branch_mode,
            max_new_tokens=config.max_new_tokens,
            response_mode=config.response_mode,
        )
        answers = [b.answer for b in gen.branches]
        clusters = _cluster(clusterer, answers, q.question)
        scores = compute_scores(gen.branches, clusters, weighting=config.entropy_weighting)
        pred = primary_answer(gen.branches, clusters)
        correct = answer_matches(pred, q.gold_answers)
        # per-branch logging so any selector (majority / confidence-weighted /
        # single) can be compared offline from the detail CSV.
        branch_logprobs = [b.mean_logprob for b in gen.branches]
        cluster_of = [0] * len(gen.branches)
        for cid, members in enumerate(clusters):
            for i in members:
                cluster_of[i] = cid
        rows.append(
            QuestionRow(
                id=q.id,
                question=q.question,
                gold_answers=q.gold_answers,
                predicted=pred,
                is_correct=correct,
                is_error=0 if correct else 1,
                scores=scores,
                generated_tokens=gen.generated_tokens,
                latency_s=gen.latency_s,
                branch_answers=answers,
                branch_logprobs=branch_logprobs,
                branch_clusters=cluster_of,
                metadata=q.metadata,
            )
        )

    return _summarise(config, rows, notes)


def _summarise(config, rows, notes) -> ExperimentResult:
    labels = [r.is_error for r in rows]
    detectors: List[DetectorSummary] = []
    branch_scores = [r.scores["branch_disagreement"] for r in rows]

    for name in SCORE_NAMES:
        col = [r.scores[name] for r in rows]
        ci = bootstrap_auc_ci(
            col, labels, n_boot=config.bootstrap_samples, seed=config.seed
        )
        summary = DetectorSummary(
            name=name, auc=ci.auc, ci_low=ci.ci_low, ci_high=ci.ci_high
        )
        if name != "branch_disagreement":
            dl = delong_test(branch_scores, col, labels)
            summary.delong_z_vs_branch = dl.z
            summary.delong_p_vs_branch = dl.p_value
        detectors.append(summary)

    n = len(rows)
    n_correct = sum(1 for r in rows if r.is_correct)
    n_error = n - n_correct
    total_tokens = sum(r.generated_tokens for r in rows)
    total_latency = sum(r.latency_s for r in rows)
    return ExperimentResult(
        config=config.to_dict(),
        rows=rows,
        detectors=detectors,
        n_questions=n,
        n_correct=n_correct,
        n_error=n_error,
        accuracy=(n_correct / n) if n else 0.0,
        total_generated_tokens=total_tokens,
        mean_tokens_per_question=(total_tokens / n) if n else 0.0,
        total_latency_s=total_latency,
        notes=notes,
    )
