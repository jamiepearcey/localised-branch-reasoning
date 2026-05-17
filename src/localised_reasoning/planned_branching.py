from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Consideration:
    label: str
    title: str
    branch_marker: str


@dataclass(frozen=True)
class BranchPlan:
    decision_point: str
    selected_factors: tuple[str, ...]
    raw_text: str
    used_default_factors: bool = False

    @property
    def branch_point(self) -> str:
        return self.decision_point

    @property
    def selected_labels(self) -> tuple[str, ...]:
        return self.selected_factors


DEFAULT_CONSIDERATIONS: tuple[Consideration, ...] = (
    Consideration(
        label="build",
        title="Implementation engineer",
        branch_marker=(
            "Give the next concrete implementation step in one normal sentence. "
            "Then write STOP_POINT: with a short reason this branch is complete. "
            "Do not repeat these instructions.\n"
        ),
    ),
    Consideration(
        label="review",
        title="Runtime reviewer",
        branch_marker=(
            "Name the likely runtime failure mode and evidence in one normal "
            "sentence. Then write STOP_POINT: with a short reason this branch is "
            "complete. Do not repeat these instructions.\n"
        ),
    ),
    Consideration(
        label="test",
        title="Regression tester",
        branch_marker=(
            "Name the smallest regression test in one normal sentence. Then write "
            "STOP_POINT: with a short reason this branch is complete. Do not repeat "
            "these instructions.\n"
        ),
    ),
)


def render_planner_prompt(
    *,
    task: str,
    considerations: tuple[Consideration, ...] = DEFAULT_CONSIDERATIONS,
) -> str:
    catalog_lines = "\n".join(
        f"- {consideration.label}: {consideration.title}"
        for consideration in considerations
    )
    allowed = ", ".join(consideration.label for consideration in considerations)
    return f"""\
You are controlling a localized-reasoning runtime.

The runtime can fork the KV cache only after you declare a decision point. For
this demo, branching is useful. Decide where to fork now, then choose at least
two and at most three factor labels from the bounded catalog below. Do not
invent labels and do not say that no branching is needed.

Allowed factors:
{catalog_lines}

Task:
{task}

Output exactly these two lines and nothing else:
DECISION_POINT: <short reason to fork now>
FACTORS: <comma-separated labels from: {allowed}>
"""


def parse_branch_plan(
    raw_text: str,
    *,
    allowed_labels: tuple[str, ...],
    default_labels: tuple[str, ...] = ("build", "review"),
) -> BranchPlan:
    decision_point = _field_value(raw_text, "DECISION_POINT") or _field_value(
        raw_text,
        "BRANCH_POINT",
    )
    if not decision_point:
        decision_point = "planner did not emit a parseable decision point"

    requested = _field_value(raw_text, "FACTORS") or _field_value(
        raw_text,
        "BRANCH_TYPES",
    )
    labels = _labels_from_text(requested, allowed_labels)
    used_default_factors = False
    if not labels:
        labels = _labels_from_text(raw_text, allowed_labels)
    if not labels:
        labels = tuple(label for label in default_labels if label in allowed_labels)
        used_default_factors = True
    if not labels:
        labels = allowed_labels[:1]
        used_default_factors = True

    return BranchPlan(
        decision_point=decision_point.strip(),
        selected_factors=_dedupe(labels),
        raw_text=raw_text,
        used_default_factors=used_default_factors,
    )


def rewrite_stacked_factors(
    plan: BranchPlan,
    *,
    considerations: tuple[Consideration, ...] = DEFAULT_CONSIDERATIONS,
) -> dict[str, str]:
    by_label = {consideration.label: consideration for consideration in considerations}
    markers: dict[str, str] = {}
    for label in plan.selected_factors:
        consideration = by_label[label]
        markers[label] = "\n" + consideration.branch_marker
    return markers


def rewrite_stacked_branch_types(
    plan: BranchPlan,
    *,
    considerations: tuple[Consideration, ...] = DEFAULT_CONSIDERATIONS,
) -> dict[str, str]:
    return rewrite_stacked_factors(plan, considerations=considerations)


def stacked_marker_for_report(plan: BranchPlan) -> str:
    return (
        "\n"
        f"Decision point: {plan.decision_point}\n"
        f"Selected factors: {', '.join(plan.selected_factors)}\n"
        "This stacked marker is for reporting only and must be rewritten into "
        "one active factor per real branch before model continuation."
    )


def _field_value(raw_text: str, field: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(raw_text)
    return match.group(1) if match else ""


def _labels_from_text(text: str, allowed_labels: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for label in allowed_labels:
        match = re.search(rf"\b{re.escape(label.lower())}\b", lowered)
        if match:
            found.append((match.start(), label))
    return tuple(label for _, label in sorted(found))


def _dedupe(labels: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return tuple(deduped)
