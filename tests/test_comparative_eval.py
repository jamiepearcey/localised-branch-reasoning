import tempfile
from pathlib import Path
import unittest

from localised_reasoning.comparative_eval import (
    AnswerScore,
    ComparativeProxyEngine,
    DEFAULT_BRANCH_REASONING_TOKENS,
    DEFAULT_SELECTOR_TOKENS,
    WorkerAnswerScorer,
    benchmark_branch_taxonomy,
    build_comparative_eval,
    compute_branch_diagnostics,
    default_benchmark_scenarios,
    default_real_world_eval_questions,
    export_comparative_eval_excel,
    fallback_taxonomy_scenarios,
    render_adjudication_prompt,
    render_answer_scoring_prompt,
    render_branch_role_question_marker,
    render_branch_taxonomy_selection_prompt,
    render_branch_reasoning_marker,
    render_meta_gate_prompt,
    render_reasoning_prompt,
    score_answer,
)
from localised_reasoning.qa_scenarios import ReasoningBudget, default_scenarios


class ComparativeEvalTest(unittest.TestCase):
    def test_scores_aliases_without_revealing_answers_to_generation(self):
        question = default_real_world_eval_questions()[1]
        everest_question = default_real_world_eval_questions()[0]

        self.assertTrue(score_answer("The ball costs five cents.", question))
        self.assertTrue(score_answer("$0.05", question))
        self.assertFalse(score_answer("$0.10", question))
        self.assertTrue(score_answer("Mount Everest.", everest_question))
        self.assertTrue(score_answer("It was still Everest.", everest_question))
        self.assertFalse(
            score_answer(
                "Before Mount Everest was identified as tallest, K2 was considered the tallest.",
                everest_question,
            )
        )

    def test_builds_win_loss_tie_comparison(self):
        scenarios = default_scenarios()
        budget = ReasoningBudget(
            branch_count=len(scenarios),
            branch_max_new_tokens=DEFAULT_BRANCH_REASONING_TOKENS,
            judge_max_new_tokens=DEFAULT_SELECTOR_TOKENS,
        )
        detail_df, summary_df, branch_df, reasoning_df = build_comparative_eval(
            scenario_engine=ComparativeProxyEngine(),
            reasoning_engine=ComparativeProxyEngine(),
            questions=default_real_world_eval_questions(),
            scenarios=scenarios,
            budget=budget,
        )

        self.assertEqual(len(detail_df), 12)
        self.assertEqual(len(branch_df), 12 * len(scenarios))
        self.assertEqual(len(reasoning_df), 12)
        self.assertIn("winner", detail_df.columns)
        self.assertIn("gated_correct", detail_df.columns)
        self.assertIn("gated_accuracy", set(summary_df["metric"]))
        self.assertGreaterEqual((detail_df["winner"] == "branch").sum(), 1)
        self.assertGreaterEqual((detail_df["winner"] == "reasoning").sum(), 1)
        self.assertGreaterEqual((detail_df["winner"] == "tie_correct").sum(), 1)
        self.assertGreaterEqual(detail_df["branch_hurt"].sum(), 1)
        self.assertIn("branch_accuracy", set(summary_df["metric"]))
        self.assertEqual(detail_df.loc[0, "reasoning_budget_tokens"], 848)

    def test_exports_comparative_workbook(self):
        detail_df, summary_df, branch_df, reasoning_df = build_comparative_eval(
            scenario_engine=ComparativeProxyEngine(),
            reasoning_engine=ComparativeProxyEngine(),
            questions=default_real_world_eval_questions()[:2],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_comparative_eval_excel(
                detail_df=detail_df,
                summary_df=summary_df,
                branch_df=branch_df,
                reasoning_df=reasoning_df,
                output_path=Path(tmpdir) / "comparison.xlsx",
            )

            self.assertTrue(output.exists())

    def test_excel_export_removes_illegal_control_characters(self):
        detail_df, summary_df, branch_df, reasoning_df = build_comparative_eval(
            scenario_engine=ComparativeProxyEngine(),
            reasoning_engine=ComparativeProxyEngine(),
            questions=default_real_world_eval_questions()[:1],
        )
        detail_df.loc[0, "question"] = "bad\x08text"

        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_comparative_eval_excel(
                detail_df=detail_df,
                summary_df=summary_df,
                branch_df=branch_df,
                reasoning_df=reasoning_df,
                output_path=Path(tmpdir) / "comparison.xlsx",
            )

            self.assertTrue(output.exists())

    def test_real_model_prompts_do_not_include_expected_answer_key(self):
        question = default_real_world_eval_questions()[1]
        budget = ReasoningBudget(
            branch_count=5,
            branch_max_new_tokens=DEFAULT_BRANCH_REASONING_TOKENS,
            judge_max_new_tokens=DEFAULT_SELECTOR_TOKENS,
        )

        adjudication_prompt = render_adjudication_prompt(
            question,
            {"fast_intuition": "$0.10.", "auditor": "$0.05."},
        )
        reasoning_prompt = render_reasoning_prompt(question, budget)

        self.assertIn(question.question, adjudication_prompt)
        self.assertIn(question.question, reasoning_prompt)
        self.assertNotIn("expected_answer", adjudication_prompt)
        self.assertNotIn("accepted_patterns", adjudication_prompt)
        self.assertNotIn(question.expected_answer, reasoning_prompt)
        self.assertIn("848", reasoning_prompt)
        self.assertIn("selector, not a solver", adjudication_prompt)

        gate_prompt = render_meta_gate_prompt(
            question,
            {"fast_intuition": "$0.10\nEVIDENCE: intuition", "auditor": "$0.05\nEVIDENCE: algebra"},
            branch_judgement=ComparativeProxyEngine().adjudicate(question, {"auditor": "$0.05"}),
            reasoning=ComparativeProxyEngine().reason_answer(question, budget),
        )
        self.assertIn("conservative answer gate", gate_prompt)
        self.assertIn("Baseline reasoning candidate", gate_prompt)
        self.assertIn("Branch diagnostics:", gate_prompt)
        self.assertIn("COLLAPSED_WEAK:", gate_prompt)
        self.assertNotIn("expected_answer", gate_prompt)
        self.assertNotIn("accepted_patterns", gate_prompt)

        scoring_prompt = render_answer_scoring_prompt(question, "$0.05")
        self.assertIn(question.expected_answer, scoring_prompt)
        self.assertIn("Given answer:", scoring_prompt)
        self.assertIn("CORRECT:", scoring_prompt)

    def test_build_can_use_custom_answer_scorer(self):
        question = default_real_world_eval_questions()[1]

        class InvertedScorer:
            name = "test-llm"

            def __init__(self):
                self.calls = []

            def score(self, question, given_answer):
                self.calls.append((question.expected_answer, given_answer))
                return AnswerScore(
                    correct=given_answer.strip() == "$0.10.",
                    confidence=91,
                    rationale="Test scorer treats the intuitive answer as correct.",
                    scorer=self.name,
                )

        scorer = InvertedScorer()
        detail_df, _, branch_df, _ = build_comparative_eval(
            scenario_engine=ComparativeProxyEngine(),
            reasoning_engine=ComparativeProxyEngine(),
            questions=[question],
            scenarios=default_scenarios()[:1],
            answer_scorer=scorer,
        )

        self.assertEqual(detail_df.loc[0, "answer_scorer"], "test-llm")
        self.assertFalse(bool(detail_df.loc[0, "branch_correct"]))
        self.assertTrue(bool(detail_df.loc[0, "reasoning_correct"]))
        self.assertTrue(bool(branch_df.loc[0, "answer_correct"]))
        self.assertTrue(all(expected == question.expected_answer for expected, _ in scorer.calls))

    def test_worker_answer_scorer_parses_llm_verdict(self):
        question = default_real_world_eval_questions()[1]

        class ScoringWorker:
            def __init__(self):
                self.requests = []

            def generate(self, prompt, **kwargs):
                self.requests.append((prompt, kwargs))
                return {
                    "text": (
                        "CORRECT: yes\n"
                        "CONFIDENCE: 94\n"
                        "RATIONALE: The given answer is equivalent to the actual answer."
                    )
                }

        worker = ScoringWorker()
        scorer = WorkerAnswerScorer(worker=worker)
        score = scorer.score(question, "$0.05")

        self.assertTrue(score.correct)
        self.assertEqual(score.confidence, 94)
        self.assertIn(question.expected_answer, worker.requests[0][0])
        self.assertEqual(worker.requests[0][1]["max_new_tokens"], 80)

    def test_branch_marker_requests_reasoning_evidence(self):
        marker = render_branch_reasoning_marker(default_scenarios()[2])

        self.assertIn("ANSWER:", marker)
        self.assertIn("CONFIDENCE:", marker)
        self.assertIn("EVIDENCE:", marker)
        self.assertIn("STOP_POINT:", marker)
        self.assertIn("trap", marker)

    def test_benchmark_scenarios_force_distinct_operations(self):
        labels = [scenario.label for scenario in default_benchmark_scenarios()]
        self.assertIn("independent_solve", labels)
        self.assertIn("eliminate_wrong_options", labels)
        self.assertIn("compute_check_only", labels)
        self.assertIn("adversarial_counterexample", labels)
        self.assertIn("source_definition_recall", labels)

        marker = render_branch_reasoning_marker(default_benchmark_scenarios()[2])
        self.assertIn("named operation", marker)
        self.assertIn("contradicts", marker)

    def test_branch_taxonomy_has_operational_roles(self):
        labels = {role.label for role in benchmark_branch_taxonomy()}

        self.assertIn("question_target_filter", labels)
        self.assertIn("formula_mapper", labels)
        self.assertIn("unit_conversion_checker", labels)
        self.assertIn("option_backsolver", labels)
        self.assertIn("evidence_answer_auditor", labels)

        physics_question = default_real_world_eval_questions()[0]
        physics_question = type(physics_question)(
            "p1",
            "mmlu-pro/physics/example",
            "How far does a muon travel?",
            "A. 4.2",
            ("A",),
        )
        selected = fallback_taxonomy_scenarios(physics_question)
        selected_labels = [scenario.label for scenario in selected]
        self.assertIn("formula_mapper", selected_labels)
        self.assertIn("unit_conversion_checker", selected_labels)

    def test_role_before_question_marker_places_operation_before_question(self):
        scenario = fallback_taxonomy_scenarios(
            type(default_real_world_eval_questions()[0])(
                "p1",
                "mmlu-pro/physics/example",
                "Question body",
                "A. answer",
                ("A",),
            )
        )[0]
        marker = render_branch_role_question_marker(scenario, "Question body")

        self.assertLess(marker.index("Branch operation:"), marker.index("Question:"))
        self.assertIn("ANSWER_LETTER:", marker)
        self.assertIn("STOP_POINT:", marker)

    def test_taxonomy_selection_prompt_does_not_include_expected_answer(self):
        question = type(default_real_world_eval_questions()[0])(
            "p1",
            "mmlu-pro/physics/example",
            "How far does a particle travel?",
            "A. hidden expected",
            ("A",),
        )
        prompt = render_branch_taxonomy_selection_prompt(
            question,
            benchmark_branch_taxonomy(),
            min_roles=3,
            max_roles=5,
        )

        self.assertIn("strict JSON", prompt)
        self.assertIn("formula_mapper", prompt)
        self.assertNotIn(question.expected_answer, prompt)

    def test_branch_diagnostics_flag_collapsed_thin_evidence(self):
        diagnostics = compute_branch_diagnostics(
            {
                "a": "B. choice\nCONFIDENCE: 80\nEVIDENCE: it matches",
                "b": "B. choice\nCONFIDENCE: 80\nEVIDENCE: best",
                "c": "B. choice\nCONFIDENCE: 80\nEVIDENCE: correct",
            }
        )

        self.assertEqual(diagnostics.unique_answer_count, 1)
        self.assertTrue(diagnostics.collapsed_weak)
        self.assertEqual(diagnostics.thin_evidence_count, 3)

    def test_option_letter_only_matches_full_option_candidate(self):
        from localised_reasoning.comparative_eval import _answers_match

        self.assertTrue(_answers_match("I", "I. Asexual reproduction in bryophytes takes place through budding"))
        self.assertTrue(_answers_match("H", "H. Regular activity"))
        self.assertFalse(_answers_match("I", "H. Regular activity"))

    def test_branch_raw_scoring_uses_selected_answer_not_evidence(self):
        question = default_real_world_eval_questions()[1]

        class EvidenceMentionsCorrectAnswer:
            def branch_answers(self, question, scenarios):
                return {
                    scenario.label: "$0.10\nCONFIDENCE: 80\nEVIDENCE: $0.05 is the algebraic result"
                    for scenario in scenarios
                }

            def adjudicate(self, question, answers):
                from localised_reasoning.qa_scenarios import JudgeResult

                return JudgeResult("$0.10", 80, "Selected the intuitive answer.")

            def reason_answer(self, question, budget):
                from localised_reasoning.qa_scenarios import ReasoningResult

                return ReasoningResult("$0.10", 80, "Baseline also selected intuition.", budget.total_tokens)

        _, _, branch_df, _ = build_comparative_eval(
            scenario_engine=EvidenceMentionsCorrectAnswer(),
            reasoning_engine=EvidenceMentionsCorrectAnswer(),
            questions=[question],
            scenarios=default_scenarios()[:1],
        )

        self.assertFalse(bool(branch_df.loc[0, "answer_correct"]))


if __name__ == "__main__":
    unittest.main()
