from pathlib import Path
import tempfile
import unittest

from localised_reasoning.qa_scenarios import (
    ReasoningBudget,
    SyntheticScenarioEngine,
    build_qa_dataframe,
    build_reasoning_dataframe,
    default_scenarios,
    default_trap_questions,
    export_qa_csv,
)


class QaScenarioPipelineTest(unittest.TestCase):
    def test_builds_one_row_per_question_with_answer_columns(self):
        questions = default_trap_questions()[:2]
        scenarios = default_scenarios()

        df = build_qa_dataframe(
            engine=SyntheticScenarioEngine(),
            questions=questions,
            scenarios=scenarios,
        )

        self.assertEqual(len(df), 2)
        self.assertIn("question", df.columns)
        self.assertIn("answer_fast_intuition", df.columns)
        self.assertIn("answer_literal_check", df.columns)
        self.assertIn("answer_skeptical_check", df.columns)
        self.assertIn("answer_contrarian_probe", df.columns)
        self.assertIn("answer_auditor", df.columns)
        self.assertIn("unique_answer_count", df.columns)
        self.assertIn("exact_match_count", df.columns)
        self.assertIn("all_branches_agree", df.columns)
        self.assertIn("final_answer", df.columns)
        self.assertIn("confidence", df.columns)

    def test_synthetic_engine_preserves_trap_resolution_in_final_answer(self):
        question = default_trap_questions()[1]

        df = build_qa_dataframe(
            engine=SyntheticScenarioEngine(),
            questions=[question],
            scenarios=default_scenarios(),
        )

        self.assertEqual(df.loc[0, "answer_fast_intuition"], "$0.10.")
        self.assertEqual(df.loc[0, "final_answer"], "$0.05.")
        self.assertGreaterEqual(df.loc[0, "unique_answer_count"], 3)
        self.assertFalse(df.loc[0, "all_branches_agree"])
        self.assertLess(df.loc[0, "confidence"], 95)

    def test_exports_csv_for_cheap_smoke_verification(self):
        df = build_qa_dataframe(
            engine=SyntheticScenarioEngine(),
            questions=default_trap_questions()[:1],
            scenarios=default_scenarios(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_qa_csv(df, Path(tmpdir) / "qa.csv")

            self.assertTrue(output.exists())
            self.assertIn("final_answer", output.read_text())

    def test_reasoning_baseline_uses_branch_plus_judge_budget(self):
        scenarios = default_scenarios()
        budget = ReasoningBudget(
            branch_count=len(scenarios),
            branch_max_new_tokens=96,
            judge_max_new_tokens=128,
        )

        df = build_reasoning_dataframe(
            engine=SyntheticScenarioEngine(),
            questions=default_trap_questions()[:2],
            budget=budget,
        )

        self.assertEqual(budget.total_tokens, 608)
        self.assertEqual(df.loc[0, "reasoning_budget_tokens"], 608)
        self.assertEqual(df.loc[0, "budget_formula"], "5 * 96 + 128")
        self.assertIn("reasoning_answer", df.columns)
        self.assertIn("reasoning_confidence", df.columns)
        self.assertIn("answer_matches_expected", df.columns)


if __name__ == "__main__":
    unittest.main()
