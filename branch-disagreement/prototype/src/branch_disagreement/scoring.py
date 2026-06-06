"""Per-question detector scores, computed from branch samples + clusters.

Every score is an **error / uncertainty** signal: higher means the answer is more
likely to be wrong. All four are computed from the *same* branches so the AUROC
comparison is paired (see ``.context/invariants.md``).

Scores:
- ``branch_disagreement``  : semantic-cluster spread (the method under test).
- ``semantic_entropy``      : Shannon entropy over semantic clusters (Kuhn et al).
- ``lexical_disagreement``  : naive self-consistency spread (exact-match clusters).
- ``neg_mean_logprob``      : negative mean token logprob (confidence baseline).
"""

import math
from collections import Counter
from typing import Dict, List, Sequence

from .model_runner import BranchSample
from .normalize import normalize_answer


SCORE_NAMES = [
    "branch_disagreement",
    "semantic_entropy",
    "lexical_disagreement",
    "neg_mean_logprob",
    "neg_logprob_single",
]


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def semantic_entropy(
    branches: Sequence[BranchSample],
    clusters: Sequence[Sequence[int]],
    weighting: str = "count",
) -> float:
    """Entropy over semantic clusters.

    ``count``: cluster probability = fraction of branches in the cluster.
    ``likelihood``: cluster probability weighted by branch sequence likelihood
    (sum of exp(mean_logprob)), normalised — closer to Kuhn et al's estimator.
    """
    n = len(branches)
    if n == 0 or not clusters:
        return 0.0
    if weighting == "likelihood":
        weights = [math.exp(b.mean_logprob) for b in branches]
        total = sum(weights) or 1.0
        probs = [sum(weights[i] for i in c) / total for c in clusters]
    else:
        probs = [len(c) / n for c in clusters]
    return _entropy(probs)


def branch_disagreement(
    branches: Sequence[BranchSample],
    clusters: Sequence[Sequence[int]],
) -> float:
    """1 - (largest semantic cluster share). 0 = total agreement."""
    n = len(branches)
    if n == 0 or not clusters:
        return 0.0
    largest = max(len(c) for c in clusters)
    return 1.0 - largest / n


def lexical_disagreement(branches: Sequence[BranchSample]) -> float:
    """1 - plurality share using exact normalized-string match. Naive baseline."""
    n = len(branches)
    if n == 0:
        return 0.0
    counts = Counter(normalize_answer(b.answer) for b in branches)
    plurality = max(counts.values())
    return 1.0 - plurality / n


def neg_mean_logprob(branches: Sequence[BranchSample]) -> float:
    """Negative mean token logprob across ALL branches. Confidence baseline.

    Higher (less confident) => more likely wrong, matching the error convention.
    Note: this uses all 8 branches, so it is NOT cheaper than the disagreement
    detectors — see ``neg_logprob_single`` for the true 1-generation baseline.
    """
    lps = [b.mean_logprob for b in branches if b.token_logprobs]
    if not lps:
        return 0.0
    return -(sum(lps) / len(lps))


def neg_logprob_single(branches: Sequence[BranchSample]) -> float:
    """Negative mean token logprob of a SINGLE generation (the first branch).

    This is the genuinely cheap baseline: one sample, 1/N the generation cost of
    the disagreement detectors. If its AUROC matches the 8-branch detectors, the
    "spend N samples to measure disagreement" premise does not pay off.
    """
    for b in branches:
        if b.token_logprobs:
            return -b.mean_logprob
    return 0.0


def primary_answer(
    branches: Sequence[BranchSample],
    clusters: Sequence[Sequence[int]],
) -> str:
    """The model's committed answer: the most common exact answer inside the
    largest semantic cluster (majority vote, semantic-cluster-aware)."""
    if not branches:
        return ""
    if not clusters:
        return branches[0].answer
    largest = max(clusters, key=lambda c: (len(c), -min(c)))
    counts = Counter(branches[i].answer for i in largest)
    return counts.most_common(1)[0][0]


def compute_scores(
    branches: Sequence[BranchSample],
    clusters: Sequence[Sequence[int]],
    weighting: str = "count",
) -> Dict[str, float]:
    return {
        "branch_disagreement": branch_disagreement(branches, clusters),
        "semantic_entropy": semantic_entropy(branches, clusters, weighting),
        "lexical_disagreement": lexical_disagreement(branches),
        "neg_mean_logprob": neg_mean_logprob(branches),
        "neg_logprob_single": neg_logprob_single(branches),
    }
