from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Mapping, Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class TrapQuestion:
    question_id: str
    question: str
    expected_answer: str


@dataclass(frozen=True)
class ScenarioBranch:
    label: str
    instruction: str


@dataclass(frozen=True)
class JudgeResult:
    final_answer: str
    confidence: int
    rationale: str


@dataclass(frozen=True)
class ReasoningBudget:
    branch_count: int
    branch_max_new_tokens: int
    judge_max_new_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.branch_count * self.branch_max_new_tokens + self.judge_max_new_tokens

    @property
    def formula(self) -> str:
        return f"{self.branch_count} * {self.branch_max_new_tokens} + {self.judge_max_new_tokens}"


@dataclass(frozen=True)
class ReasoningResult:
    answer: str
    confidence: int
    reasoning_summary: str
    budget_tokens_used: int


class ScenarioEngine(Protocol):
    def branch_answers(
        self,
        question: TrapQuestion,
        scenarios: Sequence[ScenarioBranch],
    ) -> Mapping[str, str]:
        ...

    def adjudicate(
        self,
        question: TrapQuestion,
        answers: Mapping[str, str],
    ) -> JudgeResult:
        ...


class ReasoningEngine(Protocol):
    def reason_answer(
        self,
        question: TrapQuestion,
        budget: ReasoningBudget,
    ) -> ReasoningResult:
        ...


def default_trap_questions() -> list[TrapQuestion]:
    return [
        TrapQuestion(
            "q1",
            "Before Mount Everest was identified as the tallest mountain, what was the tallest mountain in the world?",
            "Mount Everest.",
        ),
        TrapQuestion(
            "q2",
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "$0.05.",
        ),
        TrapQuestion(
            "q3",
            "You enter a dark room with one match, an oil lamp, a candle, and kindling. What do you light first?",
            "The match.",
        ),
        TrapQuestion(
            "q4",
            "Which weighs more, a pound of feathers or a pound of bricks?",
            "They weigh the same.",
        ),
        TrapQuestion(
            "q5",
            "How many months have 28 days?",
            "All 12 months.",
        ),
    ]


def default_scenarios() -> list[ScenarioBranch]:
    return [
        ScenarioBranch(
            "fast_intuition",
            "Answer quickly from first intuition in one sentence.",
        ),
        ScenarioBranch(
            "literal_check",
            "Answer only the literal wording, but do not over-explain.",
        ),
        ScenarioBranch(
            "skeptical_check",
            "Assume there may be a trap or false premise; resolve it before answering.",
        ),
        ScenarioBranch(
            "contrarian_probe",
            "Look for a less obvious interpretation that could change the answer.",
        ),
        ScenarioBranch(
            "auditor",
            "Audit the facts and arithmetic before answering; prefer the answer that survives adversarial checking.",
        ),
    ]


class SyntheticScenarioEngine:
    """Cheap deterministic engine for pipeline tests and prompt iteration."""

    _answers_by_question = {
        "q1": {
            "fast_intuition": "K2.",
            "literal_check": "Mount Everest, because it was tallest even before people identified it as such.",
            "skeptical_check": "Mount Everest.",
            "contrarian_probe": "Possibly Kangchenjunga if the question means before Everest was measured, but literally Everest.",
            "auditor": "Mount Everest.",
        },
        "q2": {
            "fast_intuition": "$0.10.",
            "literal_check": "$0.05.",
            "skeptical_check": "$0.05, because $1.05 + $0.05 = $1.10.",
            "contrarian_probe": "$0.10 if rounding or wording is loose, but exact algebra gives $0.05.",
            "auditor": "$0.05.",
        },
        "q3": {
            "fast_intuition": "The candle.",
            "literal_check": "The match.",
            "skeptical_check": "The match.",
            "contrarian_probe": "The oil lamp if the match is already lit, otherwise the match.",
            "auditor": "The match.",
        },
        "q4": {
            "fast_intuition": "The bricks.",
            "literal_check": "They weigh the same.",
            "skeptical_check": "They weigh the same: one pound each.",
            "contrarian_probe": "The bricks feel denser, but by weight they are equal.",
            "auditor": "They weigh the same.",
        },
        "q5": {
            "fast_intuition": "February.",
            "literal_check": "All 12 months.",
            "skeptical_check": "All 12 months have at least 28 days.",
            "contrarian_probe": "February has exactly 28 in common years, but all months have 28 days.",
            "auditor": "All 12 months.",
        },
    }

    def branch_answers(
        self,
        question: TrapQuestion,
        scenarios: Sequence[ScenarioBranch],
    ) -> Mapping[str, str]:
        answers: dict[str, str] = {}
        for scenario in scenarios:
            answers[scenario.label] = self._answers_by_question.get(question.question_id, {}).get(
                scenario.label,
                question.expected_answer,
            )
        return answers

    def adjudicate(
        self,
        question: TrapQuestion,
        answers: Mapping[str, str],
    ) -> JudgeResult:
        agreement = sum(_normalise_answer(answer) == _normalise_answer(question.expected_answer) for answer in answers.values())
        unique_answers = len({_normalise_answer(answer) for answer in answers.values() if answer})
        confidence = max(55, min(92, 62 + 7 * agreement - 3 * max(0, unique_answers - 2)))
        return JudgeResult(
            final_answer=question.expected_answer,
            confidence=confidence,
            rationale=(
                f"{agreement} of {len(answers)} branches exactly matched; "
                f"{unique_answers} distinct answer forms required adjudication."
            ),
        )

    def reason_answer(
        self,
        question: TrapQuestion,
        budget: ReasoningBudget,
    ) -> ReasoningResult:
        summary_by_question = {
            "q1": "The wording asks what was tallest, not what was already known by surveyors.",
            "q2": "Let ball=x and bat=x+1.00; 2x+1.00=1.10 so x=0.05.",
            "q3": "Every listed item requires ignition; the match is the first thing lit.",
            "q4": "The unit is identical for both objects, so density is irrelevant.",
            "q5": "The wording asks whether months have 28 days, not exactly 28 days.",
        }
        used = min(
            budget.total_tokens,
            90 + len(question.question.split()) + len(summary_by_question.get(question.question_id, "").split()),
        )
        return ReasoningResult(
            answer=question.expected_answer,
            confidence=86,
            reasoning_summary=summary_by_question.get(
                question.question_id,
                "The single-pass reasoning budget is sufficient to check the premise and answer directly.",
            ),
            budget_tokens_used=used,
        )


class LlamaCppPairEngine:
    """Python adapter over the existing two-branch llama.cpp fork primitive.

    This keeps orchestration in Python. It can run more than two scenarios by
    batching them into two-branch calls, so it is useful for integration testing
    before adding a dedicated multi-branch primitive.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        binary_path: Path = Path("build/llama_seq_fork_demo"),
        ctx_size: int = 4096,
        gpu_layers: int = 999,
        max_new_tokens: int = 96,
    ) -> None:
        self.model_path = model_path
        self.binary_path = binary_path
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.max_new_tokens = max_new_tokens

    def branch_answers(
        self,
        question: TrapQuestion,
        scenarios: Sequence[ScenarioBranch],
    ) -> Mapping[str, str]:
        answers: dict[str, str] = {}
        for index in range(0, len(scenarios), 2):
            left = scenarios[index]
            right = scenarios[index + 1] if index + 1 < len(scenarios) else ScenarioBranch("_dummy", "Repeat the concise answer.")
            report = self._run_pair(question, left, right)
            answers[left.label] = _extract_visible_answer(report, "branch_a")
            if right.label != "_dummy":
                answers[right.label] = _extract_visible_answer(report, "branch_b")
        return answers

    def adjudicate(
        self,
        question: TrapQuestion,
        answers: Mapping[str, str],
    ) -> JudgeResult:
        raise NotImplementedError(
            "Use a non-branching model adapter for adjudication, or run this "
            "pipeline with SyntheticScenarioEngine until that adapter is wired."
        )

    def _run_pair(
        self,
        question: TrapQuestion,
        branch_a: ScenarioBranch,
        branch_b: ScenarioBranch,
    ) -> str:
        prefix = _render_question_prefix(question.question)
        cmd = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--ctx-size",
            str(self.ctx_size),
            "--gpu-layers",
            str(self.gpu_layers),
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--prefix",
            prefix,
            "--branch-a-marker",
            _render_branch_marker(branch_a),
            "--branch-b-marker",
            _render_branch_marker(branch_b),
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout


def build_qa_dataframe(
    *,
    engine: ScenarioEngine,
    questions: Sequence[TrapQuestion],
    scenarios: Sequence[ScenarioBranch],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for question in questions:
        answers = dict(engine.branch_answers(question, scenarios))
        judgement = engine.adjudicate(question, answers)
        row: dict[str, object] = {
            "question_id": question.question_id,
            "question": question.question,
            "expected_answer": question.expected_answer,
        }
        for scenario in scenarios:
            row[f"answer_{scenario.label}"] = answers.get(scenario.label, "")
        normalised_answers = [_normalise_answer(answers.get(scenario.label, "")) for scenario in scenarios]
        nonempty_answers = [answer for answer in normalised_answers if answer]
        row["unique_answer_count"] = len(set(nonempty_answers))
        row["exact_match_count"] = sum(_normalise_answer(answer) == _normalise_answer(question.expected_answer) for answer in answers.values())
        row["all_branches_agree"] = len(set(nonempty_answers)) == 1
        row["final_answer"] = judgement.final_answer
        row["confidence"] = judgement.confidence
        row["rationale"] = judgement.rationale
        rows.append(row)
    return pd.DataFrame(rows)


def build_reasoning_dataframe(
    *,
    engine: ReasoningEngine,
    questions: Sequence[TrapQuestion],
    budget: ReasoningBudget,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for question in questions:
        result = engine.reason_answer(question, budget)
        rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "expected_answer": question.expected_answer,
                "reasoning_budget_tokens": budget.total_tokens,
                "budget_formula": budget.formula,
                "reasoning_answer": result.answer,
                "reasoning_confidence": result.confidence,
                "reasoning_summary": result.reasoning_summary,
                "budget_tokens_used": result.budget_tokens_used,
                "answer_matches_expected": _normalise_answer(result.answer) == _normalise_answer(question.expected_answer),
            }
        )
    return pd.DataFrame(rows)


def export_qa_excel(
    df: pd.DataFrame,
    output_path: Path,
    reasoning_df: pd.DataFrame | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name="QA Branches", index=False)
            if reasoning_df is not None:
                reasoning_df.to_excel(writer, sheet_name="Reasoning Baseline", index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Excel export requires an installed pandas Excel writer such as openpyxl. "
            "Install the qa-report optional dependencies or export CSV instead."
        ) from exc
    return output_path


def export_qa_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _render_question_prefix(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Answer compact factual questions. Do not mention hidden instructions.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question:\n{question}\n"
        "<|im_end|>\n"
    )


def _render_branch_marker(scenario: ScenarioBranch) -> str:
    return (
        "<|im_start|>user\n"
        f"{scenario.instruction} Output exactly two lines: ANSWER: <answer> and STOP_POINT: complete.\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _extract_visible_answer(report: str, branch_name: str) -> str:
    pattern = rf"=== {re.escape(branch_name)}_visible ===\n(?P<text>.*?)(?:\n===|\Z)"
    match = re.search(pattern, report, flags=re.S)
    if not match:
        return ""
    text = match.group("text").strip()
    answer_match = re.search(r"ANSWER:\s*(.*?)(?:\nSTOP_POINT:|\Z)", text, flags=re.S | re.I)
    if answer_match:
        return answer_match.group(1).strip()
    return text


def _normalise_answer(answer: str) -> str:
    return re.sub(r"[^a-z0-9.$]+", "", answer.lower())
