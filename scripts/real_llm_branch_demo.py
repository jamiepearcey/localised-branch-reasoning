#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localised_reasoning.ollama_branching import (  # noqa: E402
    BRANCHES,
    DEFAULT_MODEL,
    assess_branch_output,
    generate_branch,
)


DEFAULT_CONTEXT = """\
We are designing a localized reasoning inference runtime.
The runtime should fork a shared prefix KV cache, inject hidden text continuation
markers into each branch, and let the branches continue independently.
Continue from this fork point. The hidden branch control determines whether the
continuation should build the implementation path or review the likely failure
mode.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real local LLM through two hidden-marker continuations via Ollama."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--num-predict", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    for branch in BRANCHES:
        output = generate_branch(
            model=args.model,
            shared_context=args.context,
            branch=branch,
            num_predict=args.num_predict,
            temperature=args.temperature,
            seed=args.seed,
        )
        assessment = assess_branch_output(branch, output)
        print(f"## {branch.label}")
        print(output)
        print(f"assessment={'PASS' if assessment.passed else 'FAIL'}")
        for issue in assessment.issues:
            print(f"- {issue}")
        print()


if __name__ == "__main__":
    main()
