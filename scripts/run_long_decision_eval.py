#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from localised_reasoning.live_llama_worker import LiveLlamaWorker
from localised_reasoning.long_decision_eval import (
    LiveLongDecisionEngine,
    ProxyLongDecisionEngine,
    build_long_decision_eval,
    default_long_decision_cases,
    export_long_decision_eval,
    write_long_decision_csvs,
)


MODEL_PRESETS = {
    "qwen3-30b-a3b": Path("models/qwen3-30b-a3b-q4_k_m/Qwen3-30B-A3B-Q4_K_M.gguf"),
    "qwen3-14b": Path("models/qwen3-14b-q4_k_m/Qwen3-14B-Q4_K_M.gguf"),
    "qwen3-4b": Path("models/qwen3-4b-instruct-2507-q4_k_m/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
}


MODEL_DOWNLOAD_HINTS = {
    "qwen3-30b-a3b": (
        "huggingface-cli download unsloth/Qwen3-30B-A3B-GGUF "
        "Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models/qwen3-30b-a3b-q4_k_m"
    ),
    "qwen3-14b": (
        "huggingface-cli download unsloth/Qwen3-14B-GGUF "
        "Qwen3-14B-Q4_K_M.gguf --local-dir models/qwen3-14b-q4_k_m"
    ),
    "qwen3-4b": (
        "huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF "
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir models/qwen3-4b-instruct-2507-q4_k_m"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate localized KV-fork reasoning on long synthetic decision cases.")
    parser.add_argument("--engine", choices=["proxy", "worker"], default="proxy")
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default="qwen3-30b-a3b",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--worker-path", type=Path, default=Path("build/llama_branch_worker"))
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--branch-max-new-tokens", type=int, default=180)
    parser.add_argument("--planner-max-new-tokens", type=int, default=120)
    parser.add_argument("--baseline-max-new-tokens", type=int, default=700)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/long_decision_eval"))
    parser.add_argument("--output-xlsx", type=Path, default=Path("reports/long_decision_eval.xlsx"))
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    return args.model_path if args.model_path is not None else MODEL_PRESETS[args.model_preset]


def require_model_path(model_path: Path, preset: str) -> None:
    if model_path.exists():
        return
    hint = MODEL_DOWNLOAD_HINTS.get(preset)
    message = f"model file not found: {model_path}"
    if hint is not None:
        message += f"\nDownload it with:\n{hint}"
    raise SystemExit(message)


def main() -> int:
    args = parse_args()
    cases = default_long_decision_cases()[: args.limit]
    if args.engine == "proxy":
        engine = ProxyLongDecisionEngine()
        worker = None
    else:
        model_path = resolve_model_path(args)
        require_model_path(model_path, args.model_preset if args.model_path is None else "")
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
        engine = LiveLongDecisionEngine(
            worker=worker,
            branch_max_new_tokens=args.branch_max_new_tokens,
            planner_max_new_tokens=args.planner_max_new_tokens,
            baseline_max_new_tokens=args.baseline_max_new_tokens,
            request_timeout_s=args.request_timeout_s,
        )
    try:
        def progress(index, total, case) -> None:
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == total):
                print(f"progress={index}/{total} case_id={case.case_id}", file=sys.stderr, flush=True)

        case_df, branch_df, sequential_df, monolithic_df, aggregate_df, judge_df, summary_df = build_long_decision_eval(
            engine=engine,
            cases=cases,
            progress_callback=progress,
        )
    finally:
        if worker is not None:
            worker.close()

    write_long_decision_csvs(
        output_dir=args.output_dir,
        case_df=case_df,
        branch_df=branch_df,
        sequential_df=sequential_df,
        monolithic_df=monolithic_df,
        aggregate_df=aggregate_df,
        judge_df=judge_df,
        summary_df=summary_df,
    )
    output = export_long_decision_eval(
        output_xlsx=args.output_xlsx,
        case_df=case_df,
        branch_df=branch_df,
        sequential_df=sequential_df,
        monolithic_df=monolithic_df,
        aggregate_df=aggregate_df,
        judge_df=judge_df,
        summary_df=summary_df,
    )
    print(f"wrote_xlsx={output}")
    print(summary_df.to_string(index=False))
    print(case_df[["case_id", "should_branch", "selected_factors", "planner_used_fallback"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
