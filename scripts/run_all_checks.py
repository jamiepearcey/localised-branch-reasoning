#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unit, branch-marker, and marker-bias checks."
    )
    parser.add_argument(
        "--model",
        default="hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M",
        help="Ollama model used by the real-LLM checks.",
    )
    parser.add_argument(
        "--kv-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model used by the real past_key_values fork check.",
    )
    parser.add_argument("--branch-seeds", default="11,12")
    parser.add_argument("--probe-seeds", default="21,22")
    parser.add_argument("--branch-num-predict", type=int, default=180)
    parser.add_argument("--probe-num-predict", type=int, default=180)
    parser.add_argument("--kv-max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for JSON reports.",
    )
    parser.add_argument(
        "--skip-real-llm",
        action="store_true",
        help="Run only fast local unit tests.",
    )
    parser.add_argument(
        "--skip-llama-cpp-build",
        action="store_true",
        help="Skip compiling the llama.cpp sequence-fork demo.",
    )
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(str(ROOT / "src"), env.get("PYTHONPATH", ""))

    steps = [
        (
            "unit-tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        )
    ]
    if not args.skip_llama_cpp_build:
        steps.extend(
            [
                (
                    "llama-cpp-seq-fork-build",
                    ["bash", "scripts/build_llama_seq_fork_demo.sh"],
                ),
                (
                    "llama-cpp-branch-worker-build",
                    ["bash", "scripts/build_llama_branch_worker.sh"],
                ),
            ]
        )
    branch_report = args.report_dir / "branch_marker_eval.json"
    probe_report = args.report_dir / "marker_bias_probe.json"
    if not args.skip_real_llm:
        steps.extend(
            [
                (
                    "real-kv-fork",
                    [
                        sys.executable,
                        "scripts/real_kv_fork_demo.py",
                        "--model",
                        args.kv_model,
                        "--max-new-tokens",
                        str(args.kv_max_new_tokens),
                    ],
                ),
                (
                    "branch-marker-eval",
                    [
                        sys.executable,
                        "scripts/evaluate_branch_markers.py",
                        "--model",
                        args.model,
                        "--seeds",
                        args.branch_seeds,
                        "--num-predict",
                        str(args.branch_num_predict),
                        "--strict",
                        "--json-output",
                        str(branch_report),
                    ],
                ),
                (
                    "marker-bias-probe",
                    [
                        sys.executable,
                        "scripts/probe_marker_bias.py",
                        "--model",
                        args.model,
                        "--seeds",
                        args.probe_seeds,
                        "--num-predict",
                        str(args.probe_num_predict),
                        "--strict",
                        "--json-output",
                        str(probe_report),
                    ],
                ),
            ]
        )

    results: list[dict[str, object]] = []
    for name, command in steps:
        print(f"## {name}")
        started_at = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.monotonic() - started_at
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        results.append(
            {
                "name": name,
                "command": command,
                "returncode": completed.returncode,
                "elapsed_seconds": round(elapsed, 3),
            }
        )

    failed = [result for result in results if result["returncode"] != 0]
    summary = {
        "passed": not failed,
        "results": results,
        "reports": {
            "branch_marker_eval": str(branch_report) if branch_report.exists() else None,
            "marker_bias_probe": str(probe_report) if probe_report.exists() else None,
        },
    }
    summary_path = args.report_dir / "run_all_checks.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"summary={'PASS' if not failed else 'FAIL'}")
    print(f"report={summary_path}")
    if failed:
        raise SystemExit(1)


def _prepend_path(path: str, current: str) -> str:
    if not current:
        return path
    return path + os.pathsep + current


if __name__ == "__main__":
    main()
