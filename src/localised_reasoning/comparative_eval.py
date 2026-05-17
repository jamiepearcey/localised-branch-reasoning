from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol, Sequence

import pandas as pd

from localised_reasoning.live_llama_worker import LiveLlamaWorker, WorkerBranch
from localised_reasoning.qa_scenarios import (
    JudgeResult,
    ReasoningBudget,
    ReasoningEngine,
    ReasoningResult,
    ScenarioBranch,
    ScenarioEngine,
    _render_question_prefix,
    default_scenarios,
)


DEFAULT_BRANCH_REASONING_TOKENS = 160
DEFAULT_SELECTOR_TOKENS = 48
DEFAULT_META_GATE_TOKENS = 96
BRANCH_PREFIX_CACHE_ID = "branch_question_prefix_v1"
JUDGE_PREFIX_CACHE_ID = "judge_selector_prefix_v1"
META_GATE_PREFIX_CACHE_ID = "meta_gate_prefix_v1"


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    category: str
    question: str
    expected_answer: str
    accepted_patterns: tuple[str, ...]


@dataclass(frozen=True)
class MetaGateResult:
    final_answer: str
    confidence: int
    source: str
    rationale: str
    support_check: str = ""


@dataclass(frozen=True)
class AnswerScore:
    correct: bool
    confidence: int
    rationale: str
    raw_text: str = ""
    scorer: str = "regex"


@dataclass(frozen=True)
class BranchDiagnostics:
    unique_answer_count: int
    branch_count: int
    thin_evidence_count: int
    strong_evidence_count: int
    arithmetic_check_count: int
    contradiction_count: int
    answer_evidence_mismatch_count: int

    @property
    def collapsed_weak(self) -> bool:
        if self.branch_count == 0:
            return True
        if self.unique_answer_count <= 1 and self.branch_count >= 3:
            return self.strong_evidence_count < self.branch_count
        return self.unique_answer_count <= 1 and (
            self.strong_evidence_count == 0 or self.thin_evidence_count >= self.branch_count - 1
        )


class AnswerScorer(Protocol):
    name: str

    def score(self, question: EvalQuestion, given_answer: str) -> AnswerScore:
        ...


class RegexAnswerScorer:
    name = "regex"

    def score(self, question: EvalQuestion, given_answer: str) -> AnswerScore:
        correct = score_answer(given_answer, question)
        return AnswerScore(
            correct=correct,
            confidence=100 if correct else 0,
            rationale="Accepted by deterministic answer patterns." if correct else "No deterministic answer pattern matched.",
            scorer=self.name,
        )


class WorkerAnswerScorer:
    """LLM-backed scorer that sees the expected answer only after generation."""

    name = "worker-llm"

    def __init__(
        self,
        *,
        worker: LiveLlamaWorker,
        max_new_tokens: int = 80,
        timeout_s: float | None = None,
        fallback_to_regex: bool = True,
    ) -> None:
        self.worker = worker
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.fallback_to_regex = fallback_to_regex

    def score(self, question: EvalQuestion, given_answer: str) -> AnswerScore:
        regex_score = RegexAnswerScorer().score(question, given_answer)
        if _is_multiple_choice_expected(question.expected_answer) and regex_score.correct:
            return AnswerScore(
                True,
                regex_score.confidence,
                "Accepted by deterministic multiple-choice option-letter precheck.",
                scorer=f"{self.name}+mc-regex-precheck",
            )
        prompt = render_answer_scoring_prompt(question, given_answer)
        response = self.worker.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            timeout_s=self.timeout_s,
        )
        raw = _strip_empty_think_blocks(str(response.get("text", ""))).strip()
        verdict = _field_value(raw, "CORRECT").lower()
        confidence = _parse_confidence(_field_value(raw, "CONFIDENCE"))
        rationale = _field_value(raw, "RATIONALE") or raw
        if verdict.startswith(("yes", "true", "correct")):
            return AnswerScore(True, confidence, rationale, raw_text=raw, scorer=self.name)
        if verdict.startswith(("no", "false", "incorrect")):
            return AnswerScore(False, confidence, rationale, raw_text=raw, scorer=self.name)
        if not self.fallback_to_regex:
            return AnswerScore(False, confidence, f"Unparseable scorer verdict: {raw}", raw_text=raw, scorer=self.name)
        fallback = RegexAnswerScorer().score(question, given_answer)
        return AnswerScore(
            correct=fallback.correct,
            confidence=fallback.confidence,
            rationale=f"LLM scorer verdict was unparseable; fallback used. Raw: {raw}",
            raw_text=raw,
            scorer=f"{self.name}+regex-fallback",
        )


def default_real_world_eval_questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(
            "rw1",
            "counterfactual wording",
            "Before Mount Everest was identified as the tallest mountain, what was the tallest mountain in the world?",
            "Mount Everest.",
            (
                r"^(mount\s+)?everest\b",
                r"\bstill\s+(mount\s+)?everest\b",
                r"\bliterally\s+(mount\s+)?everest\b",
                r"\banswer\s+is\s+(mount\s+)?everest\b",
            ),
        ),
        EvalQuestion(
            "rw2",
            "arithmetic trap",
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "$0.05.",
            (r"\$?0\.05", "5 cents", "five cents"),
        ),
        EvalQuestion(
            "rw3",
            "literal wording",
            "You enter a dark room with one match, an oil lamp, a candle, and kindling. What do you light first?",
            "The match.",
            ("match",),
        ),
        EvalQuestion(
            "rw4",
            "programming nuance",
            "In JavaScript, what does typeof null return?",
            '"object".',
            ("object",),
        ),
        EvalQuestion(
            "rw5",
            "data format",
            "Are trailing commas valid in strict JSON?",
            "No.",
            ("no", "not valid", "invalid"),
        ),
        EvalQuestion(
            "rw6",
            "sql three-valued logic",
            "In SQL, does NULL = NULL evaluate to true?",
            "No, it evaluates to UNKNOWN/NULL rather than TRUE.",
            ("unknown", "not true", "no"),
        ),
        EvalQuestion(
            "rw7",
            "http semantics",
            "Should an HTTP 204 No Content response include a response body?",
            "No.",
            ("no", "must not", "should not", "no body"),
        ),
        EvalQuestion(
            "rw8",
            "python runtime behavior",
            "In Python, is a mutable default argument recreated on each function call?",
            "No, it is created once at function definition time.",
            ("no", "created once", "definition time", "same object"),
        ),
        EvalQuestion(
            "rw9",
            "units",
            "Which weighs more, a pound of feathers or a pound of bricks?",
            "They weigh the same.",
            ("same", "equal", "neither"),
        ),
        EvalQuestion(
            "rw10",
            "calendar wording",
            "How many months have 28 days?",
            "All 12 months.",
            ("all 12", "12", "all months"),
        ),
        EvalQuestion(
            "rw11",
            "api behavior",
            "In CSS, does display: none reserve layout space for the element?",
            "No.",
            ("no", "does not reserve", "no space"),
        ),
        EvalQuestion(
            "rw12",
            "language precision",
            "If a meeting is scheduled at 12:00 UTC, is that always noon local time for every participant?",
            "No.",
            ("no", "depends", "time zone", "not always"),
        ),
    ]


def default_benchmark_scenarios(question: EvalQuestion | None = None) -> list[ScenarioBranch]:
    return [
        ScenarioBranch(
            "independent_solve",
            "Solve independently and choose the best option before considering branch-specific checks.",
        ),
        ScenarioBranch(
            "eliminate_wrong_options",
            "Eliminate all wrong options first, then choose the survivor.",
        ),
        ScenarioBranch(
            "compute_check_only",
            "Use only calculation, formal checking, units, or exact rule application where possible.",
        ),
        ScenarioBranch(
            "adversarial_counterexample",
            "Assume the obvious answer is wrong and test the strongest counterexample or alternative.",
        ),
        ScenarioBranch(
            "source_definition_recall",
            "Recall the relevant source, definition, theorem, mechanism, rule, or fact before using options.",
        ),
    ]


class ComparativeProxyEngine:
    """Deterministic proxy that creates realistic wins, losses, and ties.

    This is not a model-quality claim. It exists so the comparative workflow can
    be inspected without paying for a large-model run.
    """

    _branch_final = {
        "rw1": ("Mount Everest.", 72, "Branch disagreement exposed the wording trap."),
        "rw2": ("$0.05.", 76, "Arithmetic and skeptical branches corrected the intuitive answer."),
        "rw3": ("The match.", 78, "Literal and auditor branches agreed on the necessary first action."),
        "rw4": ('"object".', 70, "Programming-specific branch recovered the known JavaScript quirk."),
        "rw5": ("No.", 73, "Data-format branch identified strict JSON behavior."),
        "rw6": ("TRUE, because both sides are NULL.", 71, "The final judge over-trusted the intuitive equality branch."),
        "rw7": ("It may include a small explanatory body.", 69, "The final judge over-weighted client tolerance over HTTP semantics."),
        "rw8": ("No, it is created once at function definition time.", 75, "Runtime branch caught Python's default-argument behavior."),
        "rw9": ("They weigh the same.", 70, "Literal branch ignored density and used the stated unit."),
        "rw10": ("All 12 months.", 72, "Skeptical branch caught the 'have' versus 'exactly have' distinction."),
        "rw11": ("No.", 68, "CSS branch identified that display:none removes layout participation."),
        "rw12": ("No.", 71, "Timezone branch rejected a universal local-noon interpretation."),
    }

    _reasoning = {
        "rw1": ("Mount Everest.", 86, "The question asks what was tallest, not what had been measured."),
        "rw2": ("$0.10.", 80, "The baseline used the common intuitive subtraction shortcut."),
        "rw3": ("The match.", 85, "The match must be lit before the other objects can be lit."),
        "rw4": ("null.", 79, "The baseline answered the semantic value rather than JavaScript typeof behavior."),
        "rw5": ("No.", 84, "Strict JSON does not permit trailing commas."),
        "rw6": ("No.", 70, "The baseline rejected TRUE but did not name UNKNOWN."),
        "rw7": ("No body should be included.", 82, "A 204 response communicates absence of content."),
        "rw8": ("Yes, it is recreated on each call.", 78, "The baseline overgeneralized normal local object creation."),
        "rw9": ("They weigh the same.", 86, "Both are one pound."),
        "rw10": ("February.", 82, "The baseline interpreted the question as exactly 28 days."),
        "rw11": ("No.", 84, "display:none removes the element from layout."),
        "rw12": ("No, local time depends on each participant's time zone.", 83, "UTC is a global reference, not local noon everywhere."),
    }

    _branch_answers = {
        "rw1": {
            "fast_intuition": "K2.",
            "literal_check": "Mount Everest.",
            "skeptical_check": "Mount Everest.",
            "contrarian_probe": "Kangchenjunga if asking before measurement, but literally Everest.",
            "auditor": "Mount Everest.",
        },
        "rw2": {
            "fast_intuition": "$0.10.",
            "literal_check": "$0.05.",
            "skeptical_check": "$0.05.",
            "contrarian_probe": "$0.10 if rounded loosely; exact answer is $0.05.",
            "auditor": "$0.05.",
        },
        "rw3": {
            "fast_intuition": "The candle.",
            "literal_check": "The match.",
            "skeptical_check": "The match.",
            "contrarian_probe": "Oil lamp if the match is already lit; otherwise the match.",
            "auditor": "The match.",
        },
        "rw4": {
            "fast_intuition": "null.",
            "literal_check": '"object".',
            "skeptical_check": '"object".',
            "contrarian_probe": "It should be null semantically, but typeof returns object.",
            "auditor": '"object".',
        },
        "rw5": {
            "fast_intuition": "Yes, many JavaScript parsers allow them.",
            "literal_check": "No.",
            "skeptical_check": "No, not in strict JSON.",
            "contrarian_probe": "Allowed in JSON5 but not strict JSON.",
            "auditor": "No.",
        },
        "rw6": {
            "fast_intuition": "Yes, they are both NULL.",
            "literal_check": "No.",
            "skeptical_check": "UNKNOWN.",
            "contrarian_probe": "It is NULL/UNKNOWN, not true.",
            "auditor": "UNKNOWN.",
        },
        "rw7": {
            "fast_intuition": "It can include a small body.",
            "literal_check": "No.",
            "skeptical_check": "No body should be included.",
            "contrarian_probe": "Some clients tolerate it, but semantically no.",
            "auditor": "No.",
        },
        "rw8": {
            "fast_intuition": "Yes.",
            "literal_check": "No.",
            "skeptical_check": "No, it is created once.",
            "contrarian_probe": "It behaves as shared state across calls.",
            "auditor": "No, it is created at definition time.",
        },
        "rw9": {
            "fast_intuition": "Bricks.",
            "literal_check": "They weigh the same.",
            "skeptical_check": "They weigh the same.",
            "contrarian_probe": "Bricks are denser, but the weight is equal.",
            "auditor": "They weigh the same.",
        },
        "rw10": {
            "fast_intuition": "February.",
            "literal_check": "All 12 months.",
            "skeptical_check": "All 12 months.",
            "contrarian_probe": "February exactly, all months at least.",
            "auditor": "All 12 months.",
        },
        "rw11": {
            "fast_intuition": "Yes, hidden elements can reserve space.",
            "literal_check": "No.",
            "skeptical_check": "No, display:none removes it from layout.",
            "contrarian_probe": "visibility:hidden reserves space; display:none does not.",
            "auditor": "No.",
        },
        "rw12": {
            "fast_intuition": "Yes, 12:00 means noon.",
            "literal_check": "No.",
            "skeptical_check": "No, time zones differ.",
            "contrarian_probe": "Only for participants in UTC.",
            "auditor": "No.",
        },
    }

    def branch_answers(
        self,
        question: EvalQuestion,
        scenarios: Sequence[ScenarioBranch],
    ) -> Mapping[str, str]:
        by_label = self._branch_answers.get(question.question_id)
        if by_label is None:
            return {
                scenario.label: (
                    question.expected_answer
                    if scenario.label in {"literal_check", "skeptical_check", "auditor"}
                    else "A. insufficient proxy signal"
                )
                for scenario in scenarios
            }
        return {scenario.label: by_label.get(scenario.label, "") for scenario in scenarios}

    def sample_answers(
        self,
        question: EvalQuestion,
        scenarios: Sequence[ScenarioBranch],
    ) -> Mapping[str, str]:
        return self.branch_answers(question, scenarios)

    def adjudicate(
        self,
        question: EvalQuestion,
        answers: Mapping[str, str],
    ) -> JudgeResult:
        if question.question_id not in self._branch_final:
            return JudgeResult(
                question.expected_answer,
                75,
                "Generic proxy selects the expected answer for pipeline validation only.",
            )
        answer, confidence, rationale = self._branch_final[question.question_id]
        return JudgeResult(answer, confidence, rationale)

    def reason_answer(
        self,
        question: EvalQuestion,
        budget: ReasoningBudget,
    ) -> ReasoningResult:
        if question.question_id not in self._reasoning:
            return ReasoningResult(
                question.expected_answer,
                75,
                "Generic proxy returns the expected answer for pipeline validation only.",
                min(budget.total_tokens, 160),
            )
        answer, confidence, summary = self._reasoning[question.question_id]
        used = min(budget.total_tokens, 140 + len(question.question.split()) + len(summary.split()))
        return ReasoningResult(answer, confidence, summary, used)

    def gate_answer(
        self,
        question: EvalQuestion,
        branch_answers: Mapping[str, str],
        branch_judgement: JudgeResult,
        reasoning: ReasoningResult,
    ) -> MetaGateResult:
        if branch_judgement.final_answer.strip() == reasoning.answer.strip():
            return MetaGateResult(branch_judgement.final_answer, branch_judgement.confidence, "agreement", "Branch selector and baseline agreed.", "supported")
        if branch_judgement.confidence >= reasoning.confidence + 10:
            return MetaGateResult(branch_judgement.final_answer, branch_judgement.confidence, "branch_selector", branch_judgement.rationale, "supported")
        return MetaGateResult(reasoning.answer, reasoning.confidence, "baseline_reasoning", reasoning.reasoning_summary, "fallback")


class LiveLlamaComparativeEngine:
    """Real-model comparative adapter backed by one resident llama.cpp worker."""

    def __init__(
        self,
        *,
        model_path: Path,
        worker_path: Path = Path("build/llama_branch_worker"),
        ctx_size: int = 4096,
        gpu_layers: int = 999,
        batch_size: int = 512,
        max_seqs: int = 16,
        seed: int = 1234,
        branch_max_new_tokens: int = DEFAULT_BRANCH_REASONING_TOKENS,
        judge_max_new_tokens: int = DEFAULT_SELECTOR_TOKENS,
        request_timeout_s: float | None = None,
        worker: LiveLlamaWorker | None = None,
    ) -> None:
        self.worker = worker or LiveLlamaWorker(
            model_path=model_path,
            worker_path=worker_path,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            batch_size=batch_size,
            max_seqs=max_seqs,
            seed=seed,
        )
        self.judge_max_new_tokens = judge_max_new_tokens
        self.branch_max_new_tokens = branch_max_new_tokens
        self.request_timeout_s = request_timeout_s
        self.prefix_cache_enabled = self._install_prefix_caches()

    def _install_prefix_caches(self) -> bool:
        cache_prefix = getattr(self.worker, "cache_prefix", None)
        if cache_prefix is None:
            return False
        try:
            cache_prefix(
                prefix_id=BRANCH_PREFIX_CACHE_ID,
                prefix=render_branch_static_prefix(),
                timeout_s=self.request_timeout_s,
            )
            cache_prefix(
                prefix_id=JUDGE_PREFIX_CACHE_ID,
                prefix=render_adjudication_static_prefix(),
                timeout_s=self.request_timeout_s,
            )
            cache_prefix(
                prefix_id=META_GATE_PREFIX_CACHE_ID,
                prefix=render_meta_gate_static_prefix(),
                timeout_s=self.request_timeout_s,
            )
        except RuntimeError:
            return False
        return True

    def branch_answers(
        self,
        question: EvalQuestion,
        scenarios: Sequence[ScenarioBranch],
    ) -> Mapping[str, str]:
        worker_branches = [
            WorkerBranch(label=scenario.label, marker=render_branch_reasoning_marker(scenario))
            for scenario in scenarios
        ]
        if self.prefix_cache_enabled:
            response = self.worker.cached_branch(
                prefix_id=BRANCH_PREFIX_CACHE_ID,
                suffix=render_branch_question_suffix(question.question),
                branches=worker_branches,
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                parallel=True,
                timeout_s=self.request_timeout_s,
            )
        else:
            response = self.worker.branch(
                prefix=_render_question_prefix(question.question),
                branches=worker_branches,
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                parallel=True,
                timeout_s=self.request_timeout_s,
            )
        answers: dict[str, str] = {}
        for branch in response.get("branches", []):
            label = str(branch.get("label", ""))
            text = str(branch.get("text", ""))
            answers[label] = _compact_branch_output(text)
        return {scenario.label: answers.get(scenario.label, "") for scenario in scenarios}

    def sample_answers(
        self,
        question: EvalQuestion,
        scenarios: Sequence[ScenarioBranch],
    ) -> Mapping[str, str]:
        answers: dict[str, str] = {}
        prefix = render_branch_static_prefix() + render_branch_question_suffix(question.question)
        for index, scenario in enumerate(scenarios):
            response = self.worker.generate(
                prefix + render_branch_reasoning_marker(scenario),
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                timeout_s=self.request_timeout_s,
            )
            raw = _strip_empty_think_blocks(str(response.get("text", "")))
            answers[scenario.label] = _compact_branch_output(raw)
        return {scenario.label: answers.get(scenario.label, "") for scenario in scenarios}

    def adjudicate(
        self,
        question: EvalQuestion,
        answers: Mapping[str, str],
    ) -> JudgeResult:
        if self.prefix_cache_enabled:
            response = self.worker.cached_generate(
                prefix_id=JUDGE_PREFIX_CACHE_ID,
                suffix=render_adjudication_suffix(question, answers),
                max_new_tokens=self.judge_max_new_tokens,
                timeout_s=self.request_timeout_s,
            )
        else:
            response = self.worker.generate(
                render_adjudication_prompt(question, answers),
                max_new_tokens=self.judge_max_new_tokens,
                timeout_s=self.request_timeout_s,
            )
        raw = _strip_empty_think_blocks(str(response.get("text", "")))
        final_answer = _field_value(raw, "FINAL_ANSWER")
        confidence = _parse_confidence(_field_value(raw, "CONFIDENCE"))
        rationale = _field_value(raw, "RATIONALE")
        return JudgeResult(
            final_answer=final_answer or raw.strip(),
            confidence=confidence,
            rationale=rationale,
        )

    def reason_answer(
        self,
        question: EvalQuestion,
        budget: ReasoningBudget,
    ) -> ReasoningResult:
        response = self.worker.generate(
            render_reasoning_prompt(question, budget),
            max_new_tokens=budget.total_tokens,
            timeout_s=self.request_timeout_s,
        )
        raw = _strip_empty_think_blocks(str(response.get("text", "")))
        answer = _field_value(raw, "ANSWER")
        confidence = _parse_confidence(_field_value(raw, "CONFIDENCE"))
        summary = _field_value(raw, "SUMMARY")
        return ReasoningResult(
            answer=answer or raw.strip(),
            confidence=confidence,
            reasoning_summary=summary,
            budget_tokens_used=budget.total_tokens,
        )

    def gate_answer(
        self,
        question: EvalQuestion,
        branch_answers: Mapping[str, str],
        branch_judgement: JudgeResult,
        reasoning: ReasoningResult,
    ) -> MetaGateResult:
        if self.prefix_cache_enabled:
            response = self.worker.cached_generate(
                prefix_id=META_GATE_PREFIX_CACHE_ID,
                suffix=render_meta_gate_suffix(question, branch_answers, branch_judgement, reasoning),
                max_new_tokens=DEFAULT_META_GATE_TOKENS,
                timeout_s=self.request_timeout_s,
            )
        else:
            response = self.worker.generate(
                render_meta_gate_prompt(question, branch_answers, branch_judgement, reasoning),
                max_new_tokens=DEFAULT_META_GATE_TOKENS,
                timeout_s=self.request_timeout_s,
            )
        raw = _strip_empty_think_blocks(str(response.get("text", "")))
        final_answer = _field_value(raw, "FINAL_ANSWER")
        confidence = _parse_confidence(_field_value(raw, "CONFIDENCE"))
        source = _parse_gate_source(_field_value(raw, "SOURCE"))
        support_check = _field_value(raw, "SUPPORT_CHECK")
        rationale = _field_value(raw, "RATIONALE")
        if not final_answer:
            if source == "branch_selector":
                final_answer = branch_judgement.final_answer
            else:
                final_answer = reasoning.answer
        return MetaGateResult(
            final_answer=final_answer,
            confidence=confidence,
            source=source,
            rationale=rationale or raw.strip(),
            support_check=support_check,
        )

    def close(self) -> None:
        self.worker.close()


def render_adjudication_prompt(question: EvalQuestion, answers: Mapping[str, str]) -> str:
    return render_adjudication_static_prefix() + render_adjudication_suffix(question, answers)


def render_adjudication_static_prefix() -> str:
    return (
        "<|im_start|>system\n"
        "You are a selector, not a solver. Do not solve the question from scratch. "
        "Choose the most reliable answer using only the supplied branch outputs. "
        "Do not introduce a new answer that is absent from the candidates. "
        "Reject a branch when its evidence contradicts its answer, contains an arithmetic error, "
        "or only asserts confidence without a concrete check. Majority agreement is not enough. "
        "When diagnostics say branches collapsed to one weak answer, lower confidence and say the branch set is weak.\n"
        "<|im_end|>\n"
    )


def render_adjudication_suffix(question: EvalQuestion, answers: Mapping[str, str]) -> str:
    rows = "\n".join(f"- {label}: {answer}" for label, answer in answers.items())
    diagnostics = compute_branch_diagnostics(answers)
    return (
        "<|im_start|>user\n"
        f"Question:\n{question.question}\n\n"
        f"Reasoned branch outputs:\n{rows}\n\n"
        f"Branch diagnostics:\n{render_branch_diagnostics(diagnostics)}\n\n"
        "Output exactly these four lines:\n"
        "FINAL_ANSWER: <best answer>\n"
        "CONFIDENCE: <0-100>\n"
        "SELECTED_BRANCH: <branch label>\n"
        "RATIONALE: <one short sentence about candidate reliability>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_meta_gate_prompt(
    question: EvalQuestion,
    branch_answers: Mapping[str, str],
    branch_judgement: JudgeResult,
    reasoning: ReasoningResult,
) -> str:
    return render_meta_gate_static_prefix() + render_meta_gate_suffix(
        question, branch_answers, branch_judgement, reasoning
    )


def render_meta_gate_static_prefix() -> str:
    return (
        "<|im_start|>system\n"
        "You are a conservative answer gate. Do not solve the question from scratch. "
        "Choose between the branch selector result and the baseline reasoning result. "
        "Prefer the baseline when branches collapse to weak agreement or the branch selector gives thin evidence. "
        "Prefer the branch result only when branch evidence contains a concrete calculation, domain rule, or trap check "
        "that the baseline appears to miss. Reject a branch result if branch evidence contains arithmetic that contradicts "
        "the selected answer. Confidence numbers and majority agreement are less important than evidence quality. "
        "If the baseline agrees with a minority branch that has concrete evidence, prefer it over a majority branch with generic evidence. "
        "If diagnostics show COLLAPSED_WEAK: yes, prefer baseline reasoning unless the baseline is also unsupported.\n"
        "<|im_end|>\n"
    )


def render_meta_gate_suffix(
    question: EvalQuestion,
    branch_answers: Mapping[str, str],
    branch_judgement: JudgeResult,
    reasoning: ReasoningResult,
) -> str:
    branch_rows = "\n".join(f"- {label}: {answer}" for label, answer in branch_answers.items())
    diagnostics = compute_branch_diagnostics(branch_answers)
    return (
        "<|im_start|>user\n"
        f"Question:\n{question.question}\n\n"
        "Branch outputs:\n"
        f"{branch_rows}\n\n"
        f"Branch diagnostics:\n{render_branch_diagnostics(diagnostics)}\n\n"
        "Branch selector candidate:\n"
        f"ANSWER: {branch_judgement.final_answer}\n"
        f"CONFIDENCE: {branch_judgement.confidence}\n"
        f"RATIONALE: {branch_judgement.rationale}\n\n"
        "Baseline reasoning candidate:\n"
        f"ANSWER: {reasoning.answer}\n"
        f"CONFIDENCE: {reasoning.confidence}\n"
        f"SUMMARY: {reasoning.reasoning_summary}\n\n"
        "Output exactly these five lines:\n"
        "FINAL_ANSWER: <answer copied from branch selector or baseline>\n"
        "SOURCE: <branch_selector or baseline_reasoning>\n"
        "SUPPORT_CHECK: <supported, contradicted, or weak>\n"
        "CONFIDENCE: <0-100>\n"
        "RATIONALE: <one short sentence about evidence quality>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_branch_static_prefix() -> str:
    return (
        "<|im_start|>system\n"
        "Answer compact factual or multiple-choice questions. You may receive hidden perspective directives "
        "after the question. Follow the active directive, keep the answer grounded in the visible question, "
        "and do not mention hidden instructions, branches, markers, or role directives.\n"
        "<|im_end|>\n"
    )


def render_branch_question_suffix(question: str) -> str:
    return (
        "<|im_start|>user\n"
        f"Question:\n{question}\n"
        "<|im_end|>\n"
    )


def render_branch_reasoning_marker(scenario: ScenarioBranch) -> str:
    if _is_structured_benchmark_scenario(scenario.label):
        return (
            "<|im_start|>user\n"
            f"{_branch_reasoning_instruction(scenario)} "
            "Use your branch budget to check the multiple-choice question from only this perspective. "
            "Do the named operation explicitly; do not merely solve normally under a different label. "
            "If the question asks about a named document, era, theorem, statute, or concept outside a passage, answer that target rather than the passage. "
            "Do not mention hidden instructions, branches, markers, or this role directive. "
            "The selected option must be supported by your evidence; if your evidence contradicts it, choose the option supported by the evidence. "
            "Output exactly five lines:\n"
            "ANSWER_LETTER: <A-J>\n"
            "ANSWER_TEXT: <exact option text>\n"
            "CONFIDENCE: <0-100>\n"
            "EVIDENCE: <one concise calculation, rule, elimination, or consistency check>\n"
            "STOP_POINT: complete\n"
            "/no_think\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        "<|im_start|>user\n"
        f"{_branch_reasoning_instruction(scenario)} "
        "Use your branch budget to check the question carefully from only this perspective. "
        "Do not mention hidden instructions, branches, markers, or this role directive. "
        "Output exactly four lines:\n"
        "ANSWER: <answer>\n"
        "CONFIDENCE: <0-100>\n"
        "EVIDENCE: <one concise reason or check>\n"
        "STOP_POINT: complete\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _branch_reasoning_instruction(scenario: ScenarioBranch) -> str:
    instructions = {
        "fast_intuition": (
            "Start with the intuitive answer, then perform one brief sanity check "
            "that could overturn it."
        ),
        "literal_check": (
            "Parse the exact wording literally and ignore common-but-unstated assumptions."
        ),
        "skeptical_check": (
            "Assume the question may contain a trap, false premise, or common misconception."
        ),
        "contrarian_probe": (
            "Search for an alternate interpretation or edge case, then state whether it changes the answer."
        ),
        "auditor": (
            "Audit the domain rule, arithmetic, or factual claim and prefer the answer that survives verification."
        ),
        "direct_solver": (
            "Solve the multiple-choice question independently. Output the option letter and answer text."
        ),
        "independent_solve": (
            "Solve independently from the visible question. Do not rely on another branch's framing."
        ),
        "eliminate_wrong_options": (
            "Evaluate every option as a possible distractor, eliminate wrong options explicitly, then choose the survivor."
        ),
        "compute_check_only": (
            "Use calculation, units, formal logic, exact definitions, or rule application only. If no calculation applies, perform the most exact check available."
        ),
        "adversarial_counterexample": (
            "Assume the obvious answer may be wrong. Search for a counterexample, exception, edge case, or stronger alternative before choosing."
        ),
        "source_definition_recall": (
            "Recall the relevant source, definition, theorem, mechanism, legal rule, or historical fact first, then apply it to the options."
        ),
        "option_eliminator": (
            "Eliminate incorrect options first, then choose the remaining best option. Name the decisive elimination check."
        ),
        "calculation_verifier": (
            "Verify every arithmetic step, formula, unit conversion, and substitution. Evidence must include the calculation, formula, or unit check."
        ),
        "formula_unit_check": (
            "Name the governing formula, theorem, law, or unit relation. Evidence must show the mapping from givens to that rule."
        ),
        "formula_mapper": (
            "Map the problem to the governing formula, theorem, or conservation law before doing arithmetic."
        ),
        "option_backsolver": (
            "Substitute plausible answer options back into the problem and reject options that fail the check. Evidence must name at least one rejected option."
        ),
        "distractor_eliminator": (
            "Eliminate plausible distractors before choosing. Evidence must name the error or missing condition in a wrong option."
        ),
        "source_of_truth_recall": (
            "Recall the governing fact, rule, theorem, definition, or mechanism first. Evidence must state that source-of-truth before applying it."
        ),
        "question_focus_filter": (
            "Identify the actual question target and whether the preceding passage is relevant or a distractor. Evidence must name the target source, era, or concept."
        ),
        "chronology_checker": (
            "Check the chronology, named document, and historical period before choosing. Evidence must reject at least one option from the wrong period or topic."
        ),
        "adversarial_alternative": (
            "Assume the first plausible option may be wrong. Test the strongest alternative option and choose only what the evidence supports."
        ),
        "evidence_auditor": (
            "Audit whether the stated evidence actually supports the selected option. Reject answers whose own calculation or rule contradicts them."
        ),
        "rule_elements": (
            "Identify the governing legal rule and its required elements, then match those elements to the facts."
        ),
        "exception_checker": (
            "Look specifically for exceptions, defenses, exclusions, and procedural bars that could change the answer."
        ),
        "fact_pattern_matcher": (
            "Match the legally relevant facts to each plausible option and reject options missing an element."
        ),
        "mechanism_checker": (
            "Identify the biological, clinical, or physiological mechanism that determines the answer."
        ),
        "differential_eliminator": (
            "Eliminate distractors by comparing symptoms, definitions, mechanisms, or biological constraints."
        ),
        "guideline_or_fact_check": (
            "Check the relevant guideline, anatomy, physiology, or factual constraint before choosing."
        ),
        "definition_matcher": (
            "Match the prompt to the exact definition and reject options that use nearby but incorrect concepts."
        ),
        "counterexample_tester": (
            "Test plausible answers against a concrete example, counterexample, or edge case. Evidence must include the test case."
        ),
    }
    return instructions.get(scenario.label, scenario.instruction)


def render_reasoning_prompt(question: EvalQuestion, budget: ReasoningBudget) -> str:
    return (
        "<|im_start|>system\n"
        "Answer the question with one uninterrupted reasoning attempt. "
        "Use the same total output-token budget allocated to the branching method. "
        "Check wording, traps, edge cases, domain rules, and arithmetic as needed. "
        "Do not mention branches or candidate answers.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Reasoning budget: {budget.total_tokens} output tokens ({budget.formula}).\n"
        f"Question:\n{question.question}\n\n"
        "Output exactly these three lines:\n"
        "ANSWER: <answer>\n"
        "CONFIDENCE: <0-100>\n"
        "SUMMARY: <one short sentence explaining the key reason>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_answer_scoring_prompt(question: EvalQuestion, given_answer: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are an answer-equivalence scorer, not a solver. Use the supplied "
        "expected answer as ground truth and judge only whether the given answer "
        "expresses the same answer. For multiple-choice questions, accept a "
        "matching option letter or matching option text. Mark incorrect when the "
        "given answer is missing, ambiguous, contradictory, or selects a different option. "
        "Do not solve the question from scratch.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question:\n{question.question}\n\n"
        f"Actual answer:\n{question.expected_answer}\n\n"
        f"Given answer:\n{given_answer.strip()}\n\n"
        "Output exactly these three lines:\n"
        "CORRECT: <yes or no>\n"
        "CONFIDENCE: <0-100>\n"
        "RATIONALE: <one short sentence comparing actual and given answers>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def score_answer(answer: str, question: EvalQuestion) -> bool:
    haystack = _normalise(answer)
    for pattern in question.accepted_patterns:
        if re.search(pattern, haystack, flags=re.I):
            return True
    return False


def build_comparative_eval(
    *,
    scenario_engine: ScenarioEngine,
    reasoning_engine: ReasoningEngine,
    questions: Sequence[EvalQuestion],
    scenarios: Sequence[ScenarioBranch] | None = None,
    scenario_provider: Callable[[EvalQuestion], Sequence[ScenarioBranch]] | None = None,
    budget: ReasoningBudget | None = None,
    answer_scorer: AnswerScorer | None = None,
    progress_callback: Callable[[int, int, EvalQuestion], None] | None = None,
    include_sample_baseline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = list(scenarios or default_scenarios())
    budget = budget or ReasoningBudget(
        branch_count=len(scenarios),
        branch_max_new_tokens=DEFAULT_BRANCH_REASONING_TOKENS,
        judge_max_new_tokens=DEFAULT_SELECTOR_TOKENS,
    )
    scorer = answer_scorer or RegexAnswerScorer()

    detail_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    reasoning_rows: list[dict[str, object]] = []
    for question_index, question in enumerate(questions, start=1):
        if progress_callback is not None:
            progress_callback(question_index, len(questions), question)
        question_scenarios = list(scenario_provider(question) if scenario_provider is not None else scenarios)
        branch_answers = dict(scenario_engine.branch_answers(question, question_scenarios))
        branch_diagnostics = compute_branch_diagnostics(branch_answers)
        branch_judgement = scenario_engine.adjudicate(question, branch_answers)
        sample_answers: dict[str, str] = {}
        sample_judgement: JudgeResult | None = None
        if include_sample_baseline:
            sampler = getattr(scenario_engine, "sample_answers", None)
            if sampler is not None:
                sample_answers = dict(sampler(question, question_scenarios))
                sample_judgement = scenario_engine.adjudicate(question, sample_answers)
        reasoning = reasoning_engine.reason_answer(question, budget)
        gate_answer = _gate_answer(
            scenario_engine,
            question=question,
            branch_answers=branch_answers,
            branch_judgement=branch_judgement,
            reasoning=reasoning,
        )

        score_cache: dict[str, AnswerScore] = {}

        def score_candidate(given_answer: str) -> AnswerScore:
            candidate = _candidate_answer_only(given_answer)
            key = _normalise(candidate)
            if key not in score_cache:
                score_cache[key] = scorer.score(question, candidate)
            return score_cache[key]

        branch_final_score = score_candidate(branch_judgement.final_answer)
        reasoning_score = score_candidate(reasoning.answer)
        gated_score = score_candidate(gate_answer.final_answer)
        branch_answer_scores = {
            label: score_candidate(answer)
            for label, answer in branch_answers.items()
        }
        sample_answer_scores = {
            label: score_candidate(answer)
            for label, answer in sample_answers.items()
        }
        sample_final_score = score_candidate(sample_judgement.final_answer) if sample_judgement is not None else None

        branch_final_correct = branch_final_score.correct
        reasoning_correct = reasoning_score.correct
        gated_correct = gated_score.correct
        any_branch_correct = any(score.correct for score in branch_answer_scores.values())
        any_sample_correct = any(score.correct for score in sample_answer_scores.values())
        sample_final_correct = bool(sample_final_score.correct) if sample_final_score is not None else False

        if branch_final_correct and not reasoning_correct:
            winner = "branch"
        elif reasoning_correct and not branch_final_correct:
            winner = "reasoning"
        elif branch_final_correct and reasoning_correct:
            winner = "tie_correct"
        else:
            winner = "tie_wrong"

        if gated_correct and not reasoning_correct:
            gated_winner = "gated"
        elif reasoning_correct and not gated_correct:
            gated_winner = "reasoning"
        elif gated_correct and reasoning_correct:
            gated_winner = "tie_correct"
        else:
            gated_winner = "tie_wrong"

        detail_rows.append(
            {
                "question_id": question.question_id,
                "category": question.category,
                "broad_category": _broad_benchmark_category(question.category),
                "question": question.question,
                "expected_answer": question.expected_answer,
                "branch_final_answer": branch_judgement.final_answer,
                "branch_confidence": branch_judgement.confidence,
                "branch_correct": branch_final_correct,
                "branch_score_confidence": branch_final_score.confidence,
                "branch_score_rationale": branch_final_score.rationale,
                "branch_score_raw": branch_final_score.raw_text,
                "reasoning_answer": reasoning.answer,
                "reasoning_confidence": reasoning.confidence,
                "reasoning_correct": reasoning_correct,
                "reasoning_score_confidence": reasoning_score.confidence,
                "reasoning_score_rationale": reasoning_score.rationale,
                "reasoning_score_raw": reasoning_score.raw_text,
                "gated_final_answer": gate_answer.final_answer,
                "gated_confidence": gate_answer.confidence,
                "gated_source": gate_answer.source,
                "gated_support_check": gate_answer.support_check,
                "gated_correct": gated_correct,
                "gated_score_confidence": gated_score.confidence,
                "gated_score_rationale": gated_score.rationale,
                "gated_score_raw": gated_score.raw_text,
                "gated_winner": gated_winner,
                "winner": winner,
                "any_branch_correct": any_branch_correct,
                "branch_oracle_correct": any_branch_correct,
                "branch_selector_missed_correct": any_branch_correct and not branch_final_correct,
                "branch_helped": branch_final_correct and not reasoning_correct,
                "branch_hurt": any_branch_correct and not branch_final_correct,
                "sample_final_answer": sample_judgement.final_answer if sample_judgement is not None else "",
                "sample_confidence": sample_judgement.confidence if sample_judgement is not None else 0,
                "sample_correct": sample_final_correct,
                "any_sample_correct": any_sample_correct,
                "sample_oracle_correct": any_sample_correct,
                "sample_selector_missed_correct": any_sample_correct and not sample_final_correct,
                "sample_helped": sample_final_correct and not reasoning_correct,
                "sample_hurt": any_sample_correct and not sample_final_correct,
                "gated_helped": gated_correct and not reasoning_correct,
                "gated_hurt": reasoning_correct and not gated_correct,
                "unique_branch_answer_count": branch_diagnostics.unique_answer_count,
                "branch_thin_evidence_count": branch_diagnostics.thin_evidence_count,
                "branch_strong_evidence_count": branch_diagnostics.strong_evidence_count,
                "branch_arithmetic_check_count": branch_diagnostics.arithmetic_check_count,
                "branch_contradiction_count": branch_diagnostics.contradiction_count,
                "branch_answer_evidence_mismatch_count": branch_diagnostics.answer_evidence_mismatch_count,
                "branch_collapsed_weak": branch_diagnostics.collapsed_weak,
                "reasoning_budget_tokens": budget.total_tokens,
                "budget_formula": budget.formula,
                "reasoning_tokens_used": reasoning.budget_tokens_used,
                "branch_rationale": branch_judgement.rationale,
                "reasoning_summary": reasoning.reasoning_summary,
                "gated_rationale": gate_answer.rationale,
                "answer_scorer": scorer.name,
            }
        )

        for scenario in question_scenarios:
            answer = branch_answers.get(scenario.label, "")
            answer_score = branch_answer_scores.get(scenario.label)
            if answer_score is None:
                answer_score = score_candidate(answer)
            branch_rows.append(
                {
                    "question_id": question.question_id,
                    "method": "kv_branch",
                    "scenario": scenario.label,
                    "answer": answer,
                    "answer_correct": answer_score.correct,
                    "score_confidence": answer_score.confidence,
                    "score_rationale": answer_score.rationale,
                    "score_raw": answer_score.raw_text,
                    "answer_scorer": answer_score.scorer,
                }
            )
        for scenario in question_scenarios:
            answer = sample_answers.get(scenario.label, "")
            answer_score = sample_answer_scores.get(scenario.label)
            if answer_score is None:
                continue
            branch_rows.append(
                {
                    "question_id": question.question_id,
                    "method": "independent_sample",
                    "scenario": scenario.label,
                    "answer": answer,
                    "answer_correct": answer_score.correct,
                    "score_confidence": answer_score.confidence,
                    "score_rationale": answer_score.rationale,
                    "score_raw": answer_score.raw_text,
                    "answer_scorer": answer_score.scorer,
                }
            )

        reasoning_rows.append(
            {
                "question_id": question.question_id,
                "answer": reasoning.answer,
                "confidence": reasoning.confidence,
                "answer_correct": reasoning_correct,
                "score_confidence": reasoning_score.confidence,
                "score_rationale": reasoning_score.rationale,
                "score_raw": reasoning_score.raw_text,
                "answer_scorer": reasoning_score.scorer,
                "budget_tokens_used": reasoning.budget_tokens_used,
                "reasoning_summary": reasoning.reasoning_summary,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    branch_df = pd.DataFrame(branch_rows)
    reasoning_df = pd.DataFrame(reasoning_rows)
    summary_df = build_summary(detail_df)
    return detail_df, summary_df, branch_df, reasoning_df


def export_comparative_eval_excel(
    *,
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    reasoning_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        _sanitize_excel_df(summary_df).to_excel(writer, sheet_name="Summary", index=False)
        _sanitize_excel_df(build_category_summary(detail_df)).to_excel(writer, sheet_name="By Category", index=False)
        _sanitize_excel_df(detail_df).to_excel(writer, sheet_name="Per Question", index=False)
        _sanitize_excel_df(branch_df).to_excel(writer, sheet_name="Branch Raw", index=False)
        _sanitize_excel_df(reasoning_df).to_excel(writer, sheet_name="Reasoning Raw", index=False)
    return output_path


def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    total = len(detail_df)
    branch_correct = int(detail_df["branch_correct"].sum())
    reasoning_correct = int(detail_df["reasoning_correct"].sum())
    gated_correct = int(detail_df["gated_correct"].sum())
    sample_correct = int(detail_df["sample_correct"].sum()) if "sample_correct" in detail_df else 0
    branch_oracle_correct = int(detail_df["branch_oracle_correct"].sum()) if "branch_oracle_correct" in detail_df else 0
    sample_oracle_correct = int(detail_df["sample_oracle_correct"].sum()) if "sample_oracle_correct" in detail_df else 0
    rows = [
        ("question_count", total),
        ("branch_correct", branch_correct),
        ("branch_oracle_correct", branch_oracle_correct),
        ("reasoning_correct", reasoning_correct),
        ("gated_correct", gated_correct),
        ("sample_correct", sample_correct),
        ("sample_oracle_correct", sample_oracle_correct),
        ("branch_accuracy", branch_correct / total if total else 0),
        ("branch_oracle_accuracy", branch_oracle_correct / total if total else 0),
        ("reasoning_accuracy", reasoning_correct / total if total else 0),
        ("gated_accuracy", gated_correct / total if total else 0),
        ("sample_accuracy", sample_correct / total if total else 0),
        ("sample_oracle_accuracy", sample_oracle_correct / total if total else 0),
        ("branch_selector_misses", int(detail_df["branch_selector_missed_correct"].sum()) if total and "branch_selector_missed_correct" in detail_df else 0),
        ("sample_selector_misses", int(detail_df["sample_selector_missed_correct"].sum()) if total and "sample_selector_missed_correct" in detail_df else 0),
        ("branch_only_wins", int((detail_df["winner"] == "branch").sum())),
        ("reasoning_only_wins", int((detail_df["winner"] == "reasoning").sum())),
        ("gated_only_wins", int((detail_df["gated_winner"] == "gated").sum())),
        ("reasoning_over_gate_wins", int((detail_df["gated_winner"] == "reasoning").sum())),
        ("ties_correct", int((detail_df["winner"] == "tie_correct").sum())),
        ("ties_wrong", int((detail_df["winner"] == "tie_wrong").sum())),
        ("gated_ties_correct", int((detail_df["gated_winner"] == "tie_correct").sum())),
        ("gated_ties_wrong", int((detail_df["gated_winner"] == "tie_wrong").sum())),
        ("branch_helped", int(detail_df["branch_helped"].sum())),
        ("branch_hurt", int(detail_df["branch_hurt"].sum())),
        ("gated_helped", int(detail_df["gated_helped"].sum())),
        ("gated_hurt", int(detail_df["gated_hurt"].sum())),
        ("gated_net_gain", int(detail_df["gated_helped"].sum()) - int(detail_df["gated_hurt"].sum())),
        ("sample_helped", int(detail_df["sample_helped"].sum()) if total and "sample_helped" in detail_df else 0),
        ("sample_hurt", int(detail_df["sample_hurt"].sum()) if total and "sample_hurt" in detail_df else 0),
        ("avg_unique_branch_answers", float(detail_df["unique_branch_answer_count"].mean()) if total else 0),
        ("collapsed_weak_count", int(detail_df["branch_collapsed_weak"].sum()) if total and "branch_collapsed_weak" in detail_df else 0),
        ("avg_thin_evidence_count", float(detail_df["branch_thin_evidence_count"].mean()) if total and "branch_thin_evidence_count" in detail_df else 0),
        ("avg_strong_evidence_count", float(detail_df["branch_strong_evidence_count"].mean()) if total and "branch_strong_evidence_count" in detail_df else 0),
        ("avg_arithmetic_check_count", float(detail_df["branch_arithmetic_check_count"].mean()) if total and "branch_arithmetic_check_count" in detail_df else 0),
        ("avg_answer_evidence_mismatch_count", float(detail_df["branch_answer_evidence_mismatch_count"].mean()) if total and "branch_answer_evidence_mismatch_count" in detail_df else 0),
        ("budget_tokens", int(detail_df["reasoning_budget_tokens"].iloc[0]) if total else 0),
        ("answer_scorer", str(detail_df["answer_scorer"].iloc[0]) if total and "answer_scorer" in detail_df else ""),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_category_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    df = detail_df.copy()
    if "broad_category" not in df.columns:
        df["broad_category"] = df["category"].map(_broad_benchmark_category)
    grouped = df.groupby("broad_category", dropna=False)
    rows: list[dict[str, object]] = []
    for category, group in grouped:
        total = len(group)
        rows.append(
            {
                "broad_category": category,
                "question_count": total,
                "branch_accuracy": float(group["branch_correct"].mean()) if total else 0,
                "branch_oracle_accuracy": float(group["branch_oracle_correct"].mean()) if "branch_oracle_correct" in group else 0,
                "reasoning_accuracy": float(group["reasoning_correct"].mean()) if total else 0,
                "gated_accuracy": float(group["gated_correct"].mean()) if total else 0,
                "sample_accuracy": float(group["sample_correct"].mean()) if "sample_correct" in group else 0,
                "sample_oracle_accuracy": float(group["sample_oracle_correct"].mean()) if "sample_oracle_correct" in group else 0,
                "branch_helped": int(group["branch_helped"].sum()),
                "branch_hurt": int(group["branch_hurt"].sum()),
                "branch_selector_misses": int(group["branch_selector_missed_correct"].sum()) if "branch_selector_missed_correct" in group else 0,
                "sample_selector_misses": int(group["sample_selector_missed_correct"].sum()) if "sample_selector_missed_correct" in group else 0,
                "gated_helped": int(group["gated_helped"].sum()),
                "gated_hurt": int(group["gated_hurt"].sum()),
                "gated_net_gain": int(group["gated_helped"].sum()) - int(group["gated_hurt"].sum()),
                "avg_unique_branch_answers": float(group["unique_branch_answer_count"].mean()) if total else 0,
                "collapsed_weak_count": int(group["branch_collapsed_weak"].sum()) if "branch_collapsed_weak" in group else 0,
                "avg_strong_evidence_count": float(group["branch_strong_evidence_count"].mean()) if "branch_strong_evidence_count" in group else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["question_count", "broad_category"], ascending=[False, True])


def _normalise(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip().lower())


def _sanitize_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    illegal_control_chars = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
    return df.map(
        lambda value: illegal_control_chars.sub("", value)
        if isinstance(value, str)
        else value
    )


def _field_value(raw: str, field: str) -> str:
    prefix = f"{field.lower()}:"
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix):].strip()
    return ""


def _extract_answer_text(raw: str) -> str:
    cleaned = _strip_empty_think_blocks(raw).strip()
    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?:\n(?:CONFIDENCE|EVIDENCE|STOP_POINT):|\Z)",
        cleaned,
        flags=re.S | re.I,
    )
    if answer_match:
        return answer_match.group(1).strip()
    return cleaned


def _compact_branch_output(raw: str) -> str:
    cleaned = _strip_empty_think_blocks(raw).strip()
    answer_letter = _field_value(cleaned, "ANSWER_LETTER")
    answer_text = _field_value(cleaned, "ANSWER_TEXT")
    if answer_letter and answer_text:
        answer = f"{answer_letter.strip().upper().rstrip('.')}. {answer_text.strip()}"
    elif answer_letter:
        answer = answer_letter.strip().upper().rstrip(".")
    else:
        answer = _extract_answer_text(cleaned)
    confidence = _field_value(cleaned, "CONFIDENCE")
    evidence = _field_value(cleaned, "EVIDENCE")
    parts = [answer] if answer else [cleaned]
    if confidence:
        parts.append(f"CONFIDENCE: {confidence}")
    if evidence:
        parts.append(f"EVIDENCE: {evidence}")
    return "\n".join(parts).strip()


def _candidate_answer_only(value: str) -> str:
    stripped = _strip_empty_think_blocks(value).strip()
    if "\n" in stripped:
        return stripped.splitlines()[0].strip()
    return _extract_answer_text(stripped)


def compute_branch_diagnostics(branch_answers: Mapping[str, str]) -> BranchDiagnostics:
    answers = [_candidate_answer_only(answer) for answer in branch_answers.values() if answer.strip()]
    unique_answer_count = len({_normalise(answer) for answer in answers if answer})
    thin_evidence_count = 0
    strong_evidence_count = 0
    arithmetic_check_count = 0
    contradiction_count = 0
    answer_evidence_mismatch_count = 0
    for raw in branch_answers.values():
        evidence = _branch_evidence(raw)
        if _is_thin_evidence(evidence):
            thin_evidence_count += 1
        if _is_strong_evidence(evidence):
            strong_evidence_count += 1
        if _has_arithmetic_check(evidence):
            arithmetic_check_count += 1
        if _has_contradiction_signal(evidence):
            contradiction_count += 1
        if _answer_evidence_mismatch(_candidate_answer_only(raw), evidence):
            answer_evidence_mismatch_count += 1
    return BranchDiagnostics(
        unique_answer_count=unique_answer_count,
        branch_count=len(branch_answers),
        thin_evidence_count=thin_evidence_count,
        strong_evidence_count=strong_evidence_count,
        arithmetic_check_count=arithmetic_check_count,
        contradiction_count=contradiction_count,
        answer_evidence_mismatch_count=answer_evidence_mismatch_count,
    )


def render_branch_diagnostics(diagnostics: BranchDiagnostics) -> str:
    return "\n".join(
        [
            f"UNIQUE_ANSWERS: {diagnostics.unique_answer_count}",
            f"BRANCH_COUNT: {diagnostics.branch_count}",
            f"THIN_EVIDENCE: {diagnostics.thin_evidence_count}",
            f"STRONG_EVIDENCE: {diagnostics.strong_evidence_count}",
            f"ARITHMETIC_CHECKS: {diagnostics.arithmetic_check_count}",
            f"CONTRADICTIONS: {diagnostics.contradiction_count}",
            f"ANSWER_EVIDENCE_MISMATCHES: {diagnostics.answer_evidence_mismatch_count}",
            f"COLLAPSED_WEAK: {'yes' if diagnostics.collapsed_weak else 'no'}",
        ]
    )


def _branch_evidence(raw: str) -> str:
    evidence = _field_value(raw, "EVIDENCE")
    if evidence:
        return evidence
    lines = [line for line in raw.splitlines()[1:] if not line.upper().startswith("CONFIDENCE:")]
    return " ".join(lines).strip()


def _is_thin_evidence(evidence: str) -> bool:
    words = re.findall(r"\w+", evidence)
    generic = {"because", "therefore", "clearly", "obvious", "best", "correct", "matches"}
    return len(words) < 8 or len(set(word.lower() for word in words) - generic) < 5


def _is_strong_evidence(evidence: str) -> bool:
    lowered = evidence.lower()
    concrete_markers = (
        "=", "calculate", "calculation", "formula", "unit", "substitut", "reject",
        "eliminate", "inconsistent", "fails", "rule", "definition", "mechanism",
        "counterexample", "edge case", "law", "theorem", "trace", "source", "era",
        "period", "document", "federalist", "articles of confederation", "constitution",
    )
    return len(re.findall(r"\w+", evidence)) >= 10 and any(marker in lowered for marker in concrete_markers)


def _has_arithmetic_check(evidence: str) -> bool:
    lowered = evidence.lower()
    return bool(re.search(r"\d", evidence)) and (
        bool(re.search(r"[=+\-*/×÷]", evidence))
        or any(marker in lowered for marker in ("calculate", "formula", "unit", "substitut", "ratio", "percent"))
    )


def _has_contradiction_signal(evidence: str) -> bool:
    lowered = evidence.lower()
    return any(
        marker in lowered
        for marker in ("contradict", "inconsistent", "does not support", "cannot be", "reject", "fails")
    )


def _answer_evidence_mismatch(answer: str, evidence: str) -> bool:
    answer_letter = _extract_option_letter(answer)
    selected_number = _extract_first_number(answer)
    if selected_number is not None:
        for evidence_number in _extract_result_numbers(evidence):
            tolerance = max(0.25, abs(selected_number) * 0.15)
            if abs(selected_number - evidence_number) > tolerance:
                return True
    if not answer_letter:
        return False
    lowered = evidence.lower()
    for match in re.finditer(r"\b([a-j])\b", lowered):
        letter = match.group(1).upper()
        if letter == answer_letter:
            continue
        window = lowered[max(0, match.start() - 24): match.end() + 32]
        if any(marker in window for marker in ("answer", "choose", "correct", "supports", "therefore", "so ")):
            return True
    return False


def _extract_option_letter(answer: str) -> str:
    match = re.match(r"\s*([A-Ja-j])(?:\.|\)|:|\s)", answer)
    return match.group(1).upper() if match else ""


def _answers_match(left: str, right: str) -> bool:
    left_candidate = _candidate_answer_only(left)
    right_candidate = _candidate_answer_only(right)
    if not left_candidate or not right_candidate:
        return False
    if _normalise(left_candidate) == _normalise(right_candidate):
        return True
    left_letter = _extract_option_letter(left_candidate)
    right_letter = _extract_option_letter(right_candidate)
    return bool(left_letter and right_letter and left_letter == right_letter)


def _is_multiple_choice_expected(expected_answer: str) -> bool:
    return bool(_extract_option_letter(expected_answer))


def _extract_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _extract_result_numbers(evidence: str) -> list[float]:
    numbers: list[float] = []
    for pattern in (
        r"(?:≈|=|~|about|around|roughly|result(?:s)?(?:\s+in)?|distance(?:\s+is)?)\s*([-+]?\d+(?:\.\d+)?)\s*km\b",
        r"([-+]?\d+(?:\.\d+)?)\s*km\b",
    ):
        for match in re.finditer(pattern, evidence, flags=re.I):
            try:
                numbers.append(float(match.group(1)))
            except ValueError:
                continue
    return numbers


def _gate_answer(
    engine: object,
    *,
    question: EvalQuestion,
    branch_answers: Mapping[str, str],
    branch_judgement: JudgeResult,
    reasoning: ReasoningResult,
) -> MetaGateResult:
    diagnostics = compute_branch_diagnostics(branch_answers)
    if (
        diagnostics.collapsed_weak
        and not _answers_match(branch_judgement.final_answer, reasoning.answer)
        and (diagnostics.thin_evidence_count > 0 or branch_judgement.confidence < reasoning.confidence)
    ):
        return MetaGateResult(
            final_answer=reasoning.answer,
            confidence=max(40, min(reasoning.confidence, 80)),
            source="diversity_gate_baseline",
            rationale="Branches collapsed to one weakly supported answer, so the gate preferred the baseline.",
            support_check="weak",
        )
    if not _answers_match(branch_judgement.final_answer, reasoning.answer) and _reasoning_is_supported_by_branch(branch_answers, reasoning.answer):
        return MetaGateResult(
            final_answer=reasoning.answer,
            confidence=max(50, min(reasoning.confidence, 85)),
            source="baseline_supported_by_minority_branch",
            rationale="Baseline matched a minority branch with concrete evidence, so the gate rejected the branch majority.",
            support_check="supported",
        )
    gate = getattr(engine, "gate_answer", None)
    if gate is not None:
        return gate(question, branch_answers, branch_judgement, reasoning)
    return _heuristic_gate_answer(branch_answers, branch_judgement, reasoning)


def _reasoning_is_supported_by_branch(branch_answers: Mapping[str, str], reasoning_answer: str) -> bool:
    if not _candidate_answer_only(reasoning_answer):
        return False
    for branch_answer in branch_answers.values():
        if not _answers_match(branch_answer, reasoning_answer):
            continue
        evidence = _branch_evidence(branch_answer)
        if not _is_thin_evidence(evidence):
            return True
    return False


def _heuristic_gate_answer(
    branch_answers: Mapping[str, str],
    branch_judgement: JudgeResult,
    reasoning: ReasoningResult,
) -> MetaGateResult:
    if _answers_match(branch_judgement.final_answer, reasoning.answer):
        return MetaGateResult(branch_judgement.final_answer, max(branch_judgement.confidence, reasoning.confidence), "agreement", "Branch selector and baseline agreed.")
    unique_answers = len({_normalise(answer) for answer in branch_answers.values() if answer})
    branch_evidence = " ".join(branch_answers.values()).lower()
    has_concrete_evidence = any(
        marker in branch_evidence
        for marker in ("calculate", "calculation", "equation", "rule", "law", "because", "=")
    )
    if unique_answers >= 2 and has_concrete_evidence and branch_judgement.confidence >= reasoning.confidence:
        return MetaGateResult(branch_judgement.final_answer, branch_judgement.confidence, "branch_selector", branch_judgement.rationale)
    return MetaGateResult(reasoning.answer, reasoning.confidence, "baseline_reasoning", reasoning.reasoning_summary)


def _parse_gate_source(raw: str) -> str:
    lowered = raw.strip().lower()
    if "branch" in lowered:
        return "branch_selector"
    if "agree" in lowered:
        return "agreement"
    return "baseline_reasoning"


def _is_structured_benchmark_scenario(label: str) -> bool:
    return label in {
        "direct_solver",
        "independent_solve",
        "eliminate_wrong_options",
        "compute_check_only",
        "adversarial_counterexample",
        "source_definition_recall",
        "option_eliminator",
        "calculation_verifier",
        "adversarial_alternative",
        "evidence_auditor",
        "formula_mapper",
        "formula_unit_check",
        "option_backsolver",
        "distractor_eliminator",
        "source_of_truth_recall",
        "question_focus_filter",
        "chronology_checker",
        "rule_elements",
        "exception_checker",
        "fact_pattern_matcher",
        "mechanism_checker",
        "differential_eliminator",
        "guideline_or_fact_check",
        "definition_matcher",
        "counterexample_tester",
    }


def _broad_benchmark_category(category: str) -> str:
    match = re.search(r"mmlu-pro/([^/]+)", category)
    if match:
        return match.group(1)
    return category.split("/", 1)[0] if category else "uncategorized"


def _parse_confidence(raw: str) -> int:
    match = re.search(r"\d+", raw)
    if not match:
        return 0
    return max(0, min(100, int(match.group(0))))


def _strip_empty_think_blocks(text: str) -> str:
    return re.sub(r"<think>\s*</think>\s*", "", text, flags=re.I)
