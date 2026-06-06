#!/usr/bin/env python3
"""CPU proxy smoke test: runs the whole pipeline with no heavy dependencies.

    PYTHONPATH=src python3 scripts/smoke_test.py

This is a process check, not a quality claim. It asserts the pipeline produces
finite, ordered AUROCs and that the disagreement detector beats chance on the
synthetic data (where disagreement is wired to track the latent "knows" state).
"""

import sys

from branch_disagreement.clustering import ExactMatchClusterer
from branch_disagreement.config import ExperimentConfig
from branch_disagreement.model_runner import ProxyRunner
from branch_disagreement.pipeline import run_experiment
from branch_disagreement.report import print_summary


def main() -> int:
    config = ExperimentConfig(
        engine="proxy",
        dataset="sample",
        limit=12,
        n_branches=8,
        branch_mode="self_consistency",
        bootstrap_samples=500,
        seed=0,
    )
    result = run_experiment(config, ProxyRunner(seed=config.seed), ExactMatchClusterer())
    print_summary(result)

    aucs = {d.name: d.auc for d in result.detectors}
    assert all(0.0 <= a <= 1.0 for a in aucs.values()), f"AUROC out of range: {aucs}"
    branch = aucs["branch_disagreement"]
    assert branch > 0.5, f"branch_disagreement should beat chance on proxy: {branch}"
    print(f"[smoke] OK  branch_disagreement AUROC={branch:.3f} "
          f"(proxy process check, not a quality claim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
