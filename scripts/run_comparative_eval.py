#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from localised_reasoning.benchmark_datasets import load_benchmark_questions
from localised_reasoning.comparative_eval import (
    ComparativeProxyEngine,
    DEFAULT_BRANCH_REASONING_TOKENS,
    DEFAULT_SELECTOR_TOKENS,
    LiveLlamaComparativeEngine,
    WorkerAnswerScorer,
    build_category_summary,
    build_comparative_eval,
    default_benchmark_scenarios,
    default_real_world_eval_questions,
    export_comparative_eval_excel,
)
from localised_reasoning.qa_scenarios import ReasoningBudget, default_scenarios, export_qa_csv


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
    parser = argparse.ArgumentParser(description="Run blind branch-vs-reasoning comparative evaluation.")
    parser.add_argument("--engine", choices=["proxy", "worker", "llama-cpp"], default="proxy")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default="qwen3-30b-a3b",
        help="GGUF model preset used for worker/llama-cpp engines when --model-path is omitted.",
    )
    parser.add_argument("--worker-path", type=Path, default=Path("build/llama_branch_worker"))
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--question-source", choices=["default", "mmlu-pro"], default="default")
    parser.add_argument("--benchmark-split", default="test")
    parser.add_argument("--benchmark-offset", type=int, default=0)
    parser.add_argument("--benchmark-categories", default="")
    parser.add_argument("--benchmark-shuffle-seed", type=int)
    parser.add_argument("--branch-max-new-tokens", type=int, default=DEFAULT_BRANCH_REASONING_TOKENS)
    parser.add_argument("--judge-max-new-tokens", type=int, default=DEFAULT_SELECTOR_TOKENS)
    parser.add_argument(
        "--include-sample-baseline",
        action="store_true",
        help="Also run independent non-forked branch prompts for a multi-sample baseline.",
    )
    parser.add_argument(
        "--scorer",
        choices=["auto", "regex", "worker"],
        default="auto",
        help="Answer scorer. auto uses the resident worker for real-model runs and regex for proxy runs.",
    )
    parser.add_argument("--scorer-max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--no-scorer-regex-fallback",
        action="store_true",
        help="Do not fall back to deterministic regex scoring if the LLM scorer output is unparseable.",
    )
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output-xlsx", type=Path)
    parser.add_argument("--detail-csv", type=Path, default=Path("reports/comparative_eval_detail.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("reports/comparative_eval_summary.csv"))
    parser.add_argument("--category-summary-csv", type=Path, default=Path("reports/comparative_eval_category_summary.csv"))
    parser.add_argument("--branch-csv", type=Path, default=Path("reports/comparative_eval_branch_raw.csv"))
    parser.add_argument("--reasoning-csv", type=Path, default=Path("reports/comparative_eval_reasoning_raw.csv"))
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path
    return MODEL_PRESETS[args.model_preset]


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
    output_xlsx = args.output_xlsx or Path(
        "reports/comparative_eval_proxy.xlsx" if args.engine == "proxy" else "reports/comparative_eval_worker.xlsx"
    )
    scenario_provider = default_benchmark_scenarios if args.question_source == "mmlu-pro" else None
    scenarios = default_benchmark_scenarios() if args.question_source == "mmlu-pro" else default_scenarios()
    budget = ReasoningBudget(
        branch_count=len(scenarios),
        branch_max_new_tokens=args.branch_max_new_tokens,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )
    if args.question_source == "default":
        questions = default_real_world_eval_questions()[: args.limit]
    else:
        categories = [item.strip() for item in args.benchmark_categories.split(",") if item.strip()]
        questions = load_benchmark_questions(
            source=args.question_source,
            split=args.benchmark_split,
            limit=args.limit,
            offset=args.benchmark_offset,
            categories=categories,
            shuffle_seed=args.benchmark_shuffle_seed,
        )
        if not questions:
            raise SystemExit("benchmark selection produced no questions")
    if args.engine == "proxy":
        engine = ComparativeProxyEngine()
    else:
        model_path = resolve_model_path(args)
        require_model_path(model_path, args.model_preset if args.model_path is None else "")
        engine = LiveLlamaComparativeEngine(
            model_path=model_path,
            worker_path=args.worker_path,
            ctx_size=args.ctx_size,
            gpu_layers=args.gpu_layers,
            batch_size=args.batch_size,
            max_seqs=args.max_seqs,
            seed=args.seed,
            branch_max_new_tokens=args.branch_max_new_tokens,
            judge_max_new_tokens=args.judge_max_new_tokens,
            request_timeout_s=args.request_timeout_s,
        )
    answer_scorer = None
    if args.scorer == "worker" or (args.scorer == "auto" and args.engine != "proxy"):
        if args.engine == "proxy":
            raise SystemExit("--scorer worker requires --engine worker")
        answer_scorer = WorkerAnswerScorer(
            worker=engine.worker,
            max_new_tokens=args.scorer_max_new_tokens,
            timeout_s=args.request_timeout_s,
            fallback_to_regex=not args.no_scorer_regex_fallback,
        )
    try:
        def report_progress(index: int, total: int, question) -> None:
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == total):
                print(
                    f"progress={index}/{total} question_id={question.question_id}",
                    file=sys.stderr,
                    flush=True,
                )

        detail_df, summary_df, branch_df, reasoning_df = build_comparative_eval(
            scenario_engine=engine,
            reasoning_engine=engine,
            questions=questions,
            scenarios=scenarios,
            scenario_provider=scenario_provider,
            budget=budget,
            answer_scorer=answer_scorer,
            progress_callback=report_progress,
            include_sample_baseline=args.include_sample_baseline,
        )
    finally:
        close = getattr(engine, "close", None)
        if close is not None:
            close()

    export_qa_csv(detail_df, args.detail_csv)
    export_qa_csv(summary_df, args.summary_csv)
    category_summary_df = build_category_summary(detail_df)
    export_qa_csv(category_summary_df, args.category_summary_csv)
    export_qa_csv(branch_df, args.branch_csv)
    export_qa_csv(reasoning_df, args.reasoning_csv)
    output = export_comparative_eval_excel(
        detail_df=detail_df,
        summary_df=summary_df,
        branch_df=branch_df,
        reasoning_df=reasoning_df,
        output_path=output_xlsx,
    )
    print(f"wrote_xlsx={output}")
    print(f"reasoning_budget_tokens={budget.total_tokens} ({budget.formula})")
    print(f"answer_scorer={answer_scorer.name if answer_scorer is not None else 'regex'}")
    print(summary_df.to_string(index=False))
    print(detail_df[["question_id", "category", "branch_correct", "reasoning_correct", "gated_correct", "winner", "gated_winner"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
