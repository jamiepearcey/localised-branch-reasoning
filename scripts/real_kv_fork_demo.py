#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localised_reasoning.hf_kv_branching import (  # noqa: E402
    DEFAULT_HF_MODEL,
    run_real_kv_fork,
)


DEFAULT_PREFIX = """\
You are helping design a real KV-cache branching runtime. The shared prefix
describes the goal: prefill one prefix cache, fork it once, append branch-local
tokens, and then continue visibly from each branch-local cache. Continue with
the requested role and do not mention control markers or hidden instructions.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate a real Hugging Face past_key_values KV-cache fork."
    )
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Optional transformers attention implementation, for example eager.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only models already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Write the full demo transcript to this file.",
    )
    args = parser.parse_args()
    device = None
    if args.device != "auto":
        import torch

        device = torch.device(args.device)

    result = run_real_kv_fork(
        model_name=args.model,
        prefix=args.prefix,
        branch_markers={
            "build": (
                "\nRole: implementation engineer. Output: give the next concrete "
                "engineering step. Do not mention this role directive.\nAnswer:"
            ),
            "review": (
                "\nRole: runtime reviewer. Output: name the most likely failure "
                "mode. Do not mention this role directive.\nAnswer:"
            ),
        },
        max_new_tokens=args.max_new_tokens,
        device=device,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )

    lines = [
        f"model={result.model}",
        f"device={result.device}",
        f"prefix_forward_calls={result.prefix_forward_calls}",
        f"prefix_token_count={result.prefix_token_count}",
        f"prefix_prefill_seconds={result.prefix_prefill_seconds:.3f}",
        f"fork_shared_prefix_storage={result.fork_shared_prefix_storage}",
        f"prefix_unchanged_after_branches={result.prefix_unchanged_after_branches}",
    ]
    for branch in result.branches:
        lines.extend(
            [
                f"## branch={branch.name}",
                f"marker={branch.marker!r}",
                f"marker_token_count={branch.marker_token_count}",
                f"prefix_length_before={branch.prefix_length_before}",
                f"length_after_marker={branch.length_after_marker}",
                f"length_after_visible={branch.length_after_visible}",
                (
                    "fork_vs_full_recompute_logits_max_abs_diff="
                    f"{branch.fork_logits_max_abs_diff:.8f}"
                ),
                f"suffix_checksum={branch.suffix_checksum:.4f}",
                f"visible_control_leaks={list(branch.visible_control_leaks)}",
                f"visible_token_ids={list(branch.visible_token_ids)}",
                f"visible_text={branch.visible_text!r}",
            ]
        )

    transcript = "\n".join(lines) + "\n"
    print(transcript, end="")
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(transcript, encoding="utf-8")
        print(f"output_file={args.output_file}")

    suffix_checksums = {branch.suffix_checksum for branch in result.branches}
    if len(suffix_checksums) != len(result.branches):
        raise SystemExit("branch suffix checksums did not diverge")
    if not result.fork_shared_prefix_storage:
        raise SystemExit("forks did not initially share prefix tensor storage")
    if not result.prefix_unchanged_after_branches:
        raise SystemExit("prefix cache changed after branch execution")
    if any(branch.fork_logits_max_abs_diff > 1e-3 for branch in result.branches):
        raise SystemExit("forked-cache logits diverged from full recompute")
    leaking_branches = [
        branch for branch in result.branches if branch.visible_control_leaks
    ]
    if leaking_branches:
        details = ", ".join(
            f"{branch.name}: {list(branch.visible_control_leaks)}"
            for branch in leaking_branches
        )
        raise SystemExit(f"visible output leaked control-language phrases: {details}")


if __name__ == "__main__":
    main()
