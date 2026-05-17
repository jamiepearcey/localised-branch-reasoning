from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TrainingBranch:
    branch_id: str
    factor: str
    output: str
    stop_point: str


@dataclass(frozen=True)
class BranchingTrainingRecord:
    task: str
    decision_point: str
    factors: tuple[str, ...]
    branches: tuple[TrainingBranch, ...]


def format_branching_sft_text(record: BranchingTrainingRecord) -> str:
    factors = ", ".join(record.factors)
    branch_blocks = []
    for branch in record.branches:
        branch_blocks.append(
            "\n".join(
                [
                    f"<LR_BRANCH factor=\"{branch.factor}\" id=\"{branch.branch_id}\">",
                    branch.output.strip(),
                    f"STOP_POINT: {branch.stop_point.strip()}",
                    "</LR_BRANCH>",
                ]
            )
        )
    return "\n".join(
        [
            "<LR_TASK>",
            record.task.strip(),
            "</LR_TASK>",
            f"<LR_DECISION_POINT>{record.decision_point.strip()}</LR_DECISION_POINT>",
            f"<LR_FACTORS>{factors}</LR_FACTORS>",
            *branch_blocks,
            "<LR_EVALUATE_BRANCHES/>",
        ]
    )


def parse_planned_branch_report(report_text: str) -> BranchingTrainingRecord:
    task = _section(report_text, "rewritten_shared_prefix")
    task = _between(task, "Shared task:", "Planner decision point:").strip()
    raw_plan = _section(report_text, "raw_planner_output")
    decision_point = _field(raw_plan, "DECISION_POINT")
    factors = tuple(_field(raw_plan, "FACTORS").replace(" ", "").split(","))
    factors = tuple(factor for factor in factors if factor)

    branches: list[TrainingBranch] = []
    for match in re.finditer(r"^## branch=(.+?)\n(.*?)(?=^## branch=|\Z)", report_text, re.S | re.M):
        branch_id = match.group(1).strip()
        body = match.group(2)
        output = _repr_field(body, "visible_text")
        stop_point = _repr_field(body, "model_stop_point")
        if output:
            output = _strip_stop_line(output)
        branches.append(
            TrainingBranch(
                branch_id=branch_id,
                factor=branch_id,
                output=output,
                stop_point=stop_point,
            )
        )

    if not task:
        raise ValueError("could not parse task from planned branch report")
    if not decision_point:
        raise ValueError("could not parse decision point from planned branch report")
    if not factors:
        raise ValueError("could not parse factors from planned branch report")
    if not branches:
        raise ValueError("could not parse branches from planned branch report")

    return BranchingTrainingRecord(
        task=task,
        decision_point=decision_point,
        factors=factors,
        branches=tuple(branches),
    )


def _section(text: str, name: str) -> str:
    marker = f"=== {name} ==="
    if marker not in text:
        return ""
    after = text.split(marker, 1)[1]
    next_section = re.search(r"^=== .+ ===$", after, re.M)
    return after[: next_section.start()].strip() if next_section else after.strip()


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    after = text.split(start, 1)[1]
    if end not in after:
        return after
    return after.split(end, 1)[0]


def _field(text: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+?)\s*$", text, re.M)
    return match.group(1).strip() if match else ""


def _repr_field(text: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}=(.+?)$", text, re.M)
    if not match:
        return ""
    raw = match.group(1).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return bytes(raw[1:-1], "utf-8").decode("unicode_escape")
    return raw


def _strip_stop_line(text: str) -> str:
    return re.sub(r"\s*STOP_POINT:\s*.+$", "", text, flags=re.S).strip()
