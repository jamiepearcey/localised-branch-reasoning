#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd


PRESETS = ("qwen3-4b", "qwen3-14b", "qwen3-30b-a3b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the same comparative benchmark slice across model presets.")
    parser.add_argument("--presets", default=",".join(PRESETS), help="Comma-separated model presets.")
    parser.add_argument("--question-source", choices=["default", "mmlu-pro"], default="mmlu-pro")
    parser.add_argument("--benchmark-split", default="test")
    parser.add_argument("--benchmark-offset", type=int, default=0)
    parser.add_argument("--benchmark-shuffle-seed", type=int, default=123)
    parser.add_argument("--benchmark-categories", default="")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--engine", choices=["worker", "llama-cpp"], default="worker")
    parser.add_argument("--worker-path", type=Path, default=Path("build/llama_branch_worker"))
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--scenario-mode",
        choices=["fixed", "taxonomy-category", "taxonomy-model"],
        default="fixed",
    )
    parser.add_argument(
        "--branch-layout",
        choices=["late-question-fork", "role-before-question"],
        default="late-question-fork",
        help="late-question-fork is the efficient branch-continuation architecture; role-before-question is a diagnostic accuracy probe.",
    )
    parser.add_argument("--taxonomy-min-roles", type=int, default=3)
    parser.add_argument("--taxonomy-max-roles", type=int, default=5)
    parser.add_argument("--taxonomy-selector-max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_preset_matrix"))
    parser.add_argument("--output-xlsx", type=Path, default=Path("reports/model_preset_matrix.xlsx"))
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--include-sample-baseline",
        action="store_true",
        help="Also run independent non-forked branch prompts for each preset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    presets = [preset.strip() for preset in args.presets.split(",") if preset.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    category_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for preset in presets:
        stem = f"{preset}_{args.benchmark_offset:03d}_{args.benchmark_offset + args.limit:03d}"
        detail_csv = args.output_dir / f"{stem}_detail.csv"
        summary_csv = args.output_dir / f"{stem}_summary.csv"
        category_csv = args.output_dir / f"{stem}_by_category.csv"
        command = [
            sys.executable,
            "scripts/run_comparative_eval.py",
            "--engine",
            args.engine,
            "--model-preset",
            preset,
            "--worker-path",
            str(args.worker_path),
            "--ctx-size",
            str(args.ctx_size),
            "--batch-size",
            str(args.batch_size),
            "--gpu-layers",
            str(args.gpu_layers),
            "--max-seqs",
            str(args.max_seqs),
            "--seed",
            str(args.seed),
            "--question-source",
            args.question_source,
            "--benchmark-split",
            args.benchmark_split,
            "--benchmark-offset",
            str(args.benchmark_offset),
            "--benchmark-shuffle-seed",
            str(args.benchmark_shuffle_seed),
            "--benchmark-categories",
            args.benchmark_categories,
            "--limit",
            str(args.limit),
            "--progress-every",
            str(args.progress_every),
            "--scenario-mode",
            args.scenario_mode,
            "--branch-layout",
            args.branch_layout,
            "--taxonomy-min-roles",
            str(args.taxonomy_min_roles),
            "--taxonomy-max-roles",
            str(args.taxonomy_max_roles),
            "--taxonomy-selector-max-new-tokens",
            str(args.taxonomy_selector_max_new_tokens),
            "--output-xlsx",
            str(args.output_dir / f"{stem}.xlsx"),
            "--detail-csv",
            str(detail_csv),
            "--summary-csv",
            str(summary_csv),
            "--category-summary-csv",
            str(category_csv),
            "--branch-csv",
            str(args.output_dir / f"{stem}_branch.csv"),
            "--reasoning-csv",
            str(args.output_dir / f"{stem}_reasoning.csv"),
        ]
        if args.request_timeout_s is not None:
            command.extend(["--request-timeout-s", str(args.request_timeout_s)])
        if args.include_sample_baseline:
            command.append("--include-sample-baseline")
        print(f"running_preset={preset}", file=sys.stderr, flush=True)
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            failures.append({"model_preset": preset, "returncode": str(result.returncode)})
            if not args.continue_on_error:
                return result.returncode
            continue
        detail = pd.read_csv(detail_csv)
        detail.insert(0, "model_preset", preset)
        summary = pd.read_csv(summary_csv)
        summary.insert(0, "model_preset", preset)
        category = pd.read_csv(category_csv)
        category.insert(0, "model_preset", preset)
        detail_frames.append(detail)
        summary_frames.append(summary)
        category_frames.append(category)

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    category_df = pd.concat(category_frames, ignore_index=True) if category_frames else pd.DataFrame()
    failures_df = pd.DataFrame(failures)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx) as writer:
        summary_df.to_excel(writer, sheet_name="Summary By Model", index=False)
        category_df.to_excel(writer, sheet_name="Category By Model", index=False)
        detail_df.to_excel(writer, sheet_name="Per Question", index=False)
        failures_df.to_excel(writer, sheet_name="Failures", index=False)
    print(f"wrote_xlsx={args.output_xlsx}")
    return 1 if failures and not detail_frames else 0


if __name__ == "__main__":
    raise SystemExit(main())
