from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class BranchEvaluation:
    branch_id: str
    factor: str
    stop_point: str
    relevance: int
    novelty: int
    correctness_risk: int
    actionability: int
    decision: str
    rationale: str


@dataclass(frozen=True)
class BranchResolution:
    decision: str
    selected_branch_ids: tuple[str, ...]
    merged_summary: str
    next_prompt: str


VALID_EVALUATION_DECISIONS = ("keep", "revise", "expand", "discard")
VALID_RESOLUTION_DECISIONS = ("merge", "continue_branch", "fork_branch", "replan", "stop")


def validate_branch_evaluation_payload(payload: dict[str, Any]) -> None:
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("payload must contain a non-empty evaluations list")
    for item in evaluations:
        _require_keys(
            item,
            {
                "branch_id",
                "factor",
                "stop_point",
                "relevance",
                "novelty",
                "correctness_risk",
                "actionability",
                "decision",
                "rationale",
            },
        )
        if item["decision"] not in VALID_EVALUATION_DECISIONS:
            raise ValueError(f"invalid branch decision: {item['decision']!r}")
        for key in ("relevance", "novelty", "correctness_risk", "actionability"):
            score = item[key]
            if not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"{key} must be an integer in [1, 5]")

    resolution = payload.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError("payload must contain a resolution object")
    _require_keys(
        resolution,
        {"decision", "selected_branch_ids", "merged_summary", "next_prompt"},
    )
    if resolution["decision"] not in VALID_RESOLUTION_DECISIONS:
        raise ValueError(f"invalid resolution decision: {resolution['decision']!r}")
    if not isinstance(resolution["selected_branch_ids"], list):
        raise ValueError("resolution.selected_branch_ids must be a list")


def branch_evaluation_prompt(record: dict[str, Any]) -> str:
    branches = record.get("branches", [])
    branch_lines = []
    for branch in branches:
        branch_lines.append(
            "\n".join(
                [
                    f"branch_id: {branch.get('branch_id', '')}",
                    f"factor: {branch.get('factor', '')}",
                    f"stop_point: {branch.get('stop_point', '')}",
                    f"output: {branch.get('output', '')}",
                ]
            )
        )
    return "\n\n".join(
        [
            "Evaluate completed localized-reasoning branches.",
            f"task: {record.get('task', '')}",
            f"decision_point: {record.get('decision_point', '')}",
            "branches:",
            "\n---\n".join(branch_lines),
            (
                "Return JSON with evaluations[] and resolution. Scores are 1-5. "
                f"Branch decisions: {', '.join(VALID_EVALUATION_DECISIONS)}. "
                f"Resolution decisions: {', '.join(VALID_RESOLUTION_DECISIONS)}."
            ),
        ]
    )


def payload_to_json(payload: dict[str, Any]) -> str:
    validate_branch_evaluation_payload(payload)
    return json.dumps(payload, sort_keys=True)


def _require_keys(item: dict[str, Any], keys: set[str]) -> None:
    missing = keys - set(item)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
