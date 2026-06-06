"""Pure-stdlib detection metrics: AUROC, bootstrap CI, and the DeLong test.

No numpy / scipy. The label convention throughout is **positive = error**: a
detector score should be *higher* when the answer is more likely to be wrong, so
AUROC measures how well the score ranks wrong answers above correct ones.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def _midranks(values: Sequence[float]) -> List[float]:
    """1-based midranks (average rank within tie groups)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0  # 1-based average rank
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC via the Mann-Whitney statistic, tie-aware.

    ``labels`` are 0/1 with 1 = positive (error). Returns 0.5 when one class is
    empty (undefined, treated as chance).
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _midranks(list(scores))
    sum_pos_ranks = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class AUCResult:
    auc: float
    ci_low: float
    ci_high: float
    n: int
    n_pos: int


def bootstrap_auc_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> AUCResult:
    """Percentile bootstrap CI for AUROC. Deterministic given ``seed``."""
    auc = roc_auc(scores, labels)
    n = len(scores)
    n_pos = sum(1 for y in labels if y == 1)
    if n_pos == 0 or n_pos == n or n_boot <= 0:
        return AUCResult(auc, auc, auc, n, n_pos)
    rng = random.Random(seed)
    boot = []
    idx = range(n)
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in idx]
        bs = [scores[i] for i in sample]
        bl = [labels[i] for i in sample]
        if 0 < sum(bl) < n:  # skip degenerate resamples
            boot.append(roc_auc(bs, bl))
    if not boot:
        return AUCResult(auc, auc, auc, n, n_pos)
    boot.sort()
    lo = boot[max(0, int((alpha / 2) * len(boot)))]
    hi = boot[min(len(boot) - 1, int((1 - alpha / 2) * len(boot)))]
    return AUCResult(auc, lo, hi, n, n_pos)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _cov2(a: Sequence[float], b: Sequence[float]) -> float:
    """Sample covariance (ddof=1) of two equal-length sequences."""
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n - 1)


@dataclass
class DeLongResult:
    auc_a: float
    auc_b: float
    diff: float
    z: float
    p_value: float


def delong_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[int],
) -> DeLongResult:
    """DeLong test for two correlated AUROCs on the same labels.

    Fast DeLong (Sun & Xu, 2014) specialised to two predictors, pure Python.
    ``scores_a`` is the detector of interest, ``scores_b`` the baseline; the test
    is two-sided. A non-significant p-value with auc_a > auc_b is the desired
    "competitive at lower cost" outcome for a baseline like semantic entropy.
    """
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    m, n = len(pos_idx), len(neg_idx)
    if m == 0 or n == 0:
        return DeLongResult(0.5, 0.5, 0.0, 0.0, 1.0)

    def components(scores):
        x = [scores[i] for i in pos_idx]   # positives
        y = [scores[i] for i in neg_idx]   # negatives
        tx = _midranks(x)
        ty = _midranks(y)
        tz = _midranks(x + y)
        tz_pos = tz[:m]
        tz_neg = tz[m:]
        auc = sum(tz_pos) / (m * n) - (m + 1.0) / (2.0 * n)
        v01 = [(tz_pos[i] - tx[i]) / n for i in range(m)]      # over positives
        v10 = [1.0 - (tz_neg[j] - ty[j]) / m for j in range(n)]  # over negatives
        return auc, v01, v10

    auc_a, v01a, v10a = components(scores_a)
    auc_b, v01b, v10b = components(scores_b)

    sx = _cov2(v01a, v01a), _cov2(v01a, v01b), _cov2(v01b, v01b)
    sy = _cov2(v10a, v10a), _cov2(v10a, v10b), _cov2(v10b, v10b)
    # delong covariance = sx/m + sy/n  (2x2, stored as (aa, ab, bb))
    c_aa = sx[0] / m + sy[0] / n
    c_ab = sx[1] / m + sy[1] / n
    c_bb = sx[2] / m + sy[2] / n
    var = c_aa + c_bb - 2.0 * c_ab
    diff = auc_a - auc_b
    if var <= 0.0:
        z = 0.0 if diff == 0.0 else math.copysign(float("inf"), diff)
        p = 1.0 if diff == 0.0 else 0.0
        return DeLongResult(auc_a, auc_b, diff, z, p)
    z = diff / math.sqrt(var)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return DeLongResult(auc_a, auc_b, diff, z, p)
