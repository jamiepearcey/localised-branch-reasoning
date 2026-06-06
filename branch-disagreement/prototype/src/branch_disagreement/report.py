"""Write experiment results to disk (stdlib csv/json) and print a summary."""

import csv
import json
import os
from typing import List

from .pipeline import ExperimentResult


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_detail_csv(result: ExperimentResult, path: str) -> None:
    score_names = [d.name for d in result.detectors]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "question", "predicted", "gold_answers", "is_correct",
             "is_error", "generated_tokens", "latency_s"]
            + score_names
            + ["branch_answers", "branch_logprobs", "branch_clusters"]
        )
        for r in result.rows:
            w.writerow(
                [r.id, r.question, r.predicted, " | ".join(r.gold_answers),
                 int(r.is_correct), r.is_error, r.generated_tokens,
                 round(r.latency_s, 4)]
                + [round(r.scores[n], 4) for n in score_names]
                + [" | ".join(r.branch_answers),
                   " | ".join(f"{x:.4f}" for x in r.branch_logprobs),
                   " | ".join(str(c) for c in r.branch_clusters)]
            )


def write_summary_csv(result: ExperimentResult, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["detector", "auc", "ci_low", "ci_high",
             "delong_z_vs_branch", "delong_p_vs_branch"]
        )
        for d in result.detectors:
            w.writerow(
                [d.name, round(d.auc, 4), round(d.ci_low, 4), round(d.ci_high, 4),
                 "" if d.delong_z_vs_branch is None else round(d.delong_z_vs_branch, 4),
                 "" if d.delong_p_vs_branch is None else round(d.delong_p_vs_branch, 4)]
            )


def write_json(result: ExperimentResult, path: str) -> None:
    payload = {
        "config": result.config,
        "n_questions": result.n_questions,
        "n_correct": result.n_correct,
        "n_error": result.n_error,
        "accuracy": result.accuracy,
        "total_generated_tokens": result.total_generated_tokens,
        "mean_tokens_per_question": result.mean_tokens_per_question,
        "total_latency_s": result.total_latency_s,
        "notes": result.notes,
        "detectors": [
            {
                "name": d.name,
                "auc": d.auc,
                "ci_low": d.ci_low,
                "ci_high": d.ci_high,
                "delong_z_vs_branch": d.delong_z_vs_branch,
                "delong_p_vs_branch": d.delong_p_vs_branch,
            }
            for d in result.detectors
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_all(result: ExperimentResult, reports_dir: str, prefix: str) -> List[str]:
    ensure_dir(reports_dir)
    detail = os.path.join(reports_dir, f"{prefix}_detail.csv")
    summary = os.path.join(reports_dir, f"{prefix}_summary.csv")
    js = os.path.join(reports_dir, f"{prefix}.json")
    write_detail_csv(result, detail)
    write_summary_csv(result, summary)
    write_json(result, js)
    return [detail, summary, js]


def print_summary(result: ExperimentResult) -> None:
    cfg = result.config
    print()
    print("=" * 72)
    print(f"engine={cfg['engine']}  dataset={cfg['dataset']}  model={cfg['model']}")
    print(f"branch_mode={cfg['branch_mode']}  n_branches={cfg['n_branches']}  "
          f"temperature={cfg['temperature']}")
    print("-" * 72)
    print(f"questions={result.n_questions}  accuracy={result.accuracy:.3f}  "
          f"errors={result.n_error}")
    print(f"generated tokens/question (cost axis) = "
          f"{result.mean_tokens_per_question:.1f}")
    if result.total_latency_s:
        print(f"total latency = {result.total_latency_s:.1f}s")
    for note in result.notes:
        print(f"note: {note}")
    print("-" * 72)
    print(f"{'detector':<22}{'AUROC':>8}  {'95% CI':>16}   {'DeLong p vs branch':>18}")
    for d in result.detectors:
        ci = f"[{d.ci_low:.3f},{d.ci_high:.3f}]"
        p = "" if d.delong_p_vs_branch is None else f"{d.delong_p_vs_branch:.3f}"
        print(f"{d.name:<22}{d.auc:>8.3f}  {ci:>16}   {p:>18}")
    print("=" * 72)
    print("Reminder: positive class = error. Higher AUROC = better hallucination")
    print("detection. Report AUROC together with the cost axis above.")
    print()
