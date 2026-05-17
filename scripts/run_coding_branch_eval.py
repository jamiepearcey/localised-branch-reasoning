#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from localised_reasoning.coding_branch_eval import (
    LiveCodingBranchEngine,
    ProxyCodingBranchEngine,
    build_coding_branch_eval,
    default_coding_cases,
    export_coding_branch_eval,
    write_coding_branch_csvs,
)
from localised_reasoning.live_llama_worker import LiveLlamaWorker


MODEL_PRESETS = {
    "qwen3-30b-a3b": Path("models/qwen3-30b-a3b-q4_k_m/Qwen3-30B-A3B-Q4_K_M.gguf"),
    "qwen3-14b": Path("models/qwen3-14b-q4_k_m/Qwen3-14B-Q4_K_M.gguf"),
    "qwen3-4b": Path("models/qwen3-4b-instruct-2507-q4_k_m/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-checkpoint coding branch/collapse reasoning.")
    parser.add_argument("--engine", choices=["proxy", "worker"], default="proxy")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="qwen3-30b-a3b")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--worker-path", type=Path, default=Path("build/llama_branch_worker"))
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--consideration-limit", type=int, default=5)
    parser.add_argument("--branch-max-new-tokens", type=int, default=150)
    parser.add_argument("--collapse-max-new-tokens", type=int, default=220)
    parser.add_argument("--final-max-new-tokens", type=int, default=500)
    parser.add_argument("--planner-max-new-tokens", type=int, default=120)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/coding_branch_eval"))
    parser.add_argument("--output-xlsx", type=Path, default=Path("reports/coding_branch_eval.xlsx"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = default_coding_cases()[: args.limit]
    worker = None
    if args.engine == "proxy":
        engine = ProxyCodingBranchEngine()
    else:
        model_path = args.model_path or MODEL_PRESETS[args.model_preset]
        if not model_path.exists():
            raise SystemExit(f"model file not found: {model_path}")
        worker = LiveLlamaWorker(
            model_path=model_path,
            worker_path=args.worker_path,
            ctx_size=args.ctx_size,
            gpu_layers=args.gpu_layers,
            batch_size=args.batch_size,
            max_seqs=args.max_seqs,
            seed=args.seed,
            startup_timeout_s=args.request_timeout_s or 180.0,
        )
        engine = LiveCodingBranchEngine(
            worker=worker,
            branch_max_new_tokens=args.branch_max_new_tokens,
            collapse_max_new_tokens=args.collapse_max_new_tokens,
            final_max_new_tokens=args.final_max_new_tokens,
            planner_max_new_tokens=args.planner_max_new_tokens,
            request_timeout_s=args.request_timeout_s,
        )
    try:
        def progress(index, total, case) -> None:
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == total):
                print(f"progress={index}/{total} case_id={case.case_id}", file=sys.stderr, flush=True)

        case_df, branch_df, collapse_df, final_df, monolithic_df, summary_df = build_coding_branch_eval(
            engine=engine,
            cases=cases,
            consideration_limit=args.consideration_limit,
            progress_callback=progress,
        )
    finally:
        if worker is not None:
            worker.close()

    write_coding_branch_csvs(
        output_dir=args.output_dir,
        case_df=case_df,
        branch_df=branch_df,
        collapse_df=collapse_df,
        final_df=final_df,
        monolithic_df=monolithic_df,
        summary_df=summary_df,
    )
    output = export_coding_branch_eval(
        output_xlsx=args.output_xlsx,
        case_df=case_df,
        branch_df=branch_df,
        collapse_df=collapse_df,
        final_df=final_df,
        monolithic_df=monolithic_df,
        summary_df=summary_df,
    )
    print(f"wrote_xlsx={output}")
    print(summary_df.to_string(index=False))
    print(case_df[["case_id", "selected_checkpoints", "expected_hit_rate", "planner_used_fallback"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
