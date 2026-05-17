from __future__ import annotations

import unittest

from localised_reasoning.planned_branching import (
    DEFAULT_CONSIDERATIONS,
    parse_branch_plan,
    render_planner_prompt,
    rewrite_stacked_factors,
    stacked_marker_for_report,
)


class PlannedBranchingTests(unittest.TestCase):
    def test_parse_branch_plan_extracts_allowed_factors(self) -> None:
        plan = parse_branch_plan(
            "DECISION_POINT: need independent views\nFACTORS: review, build, unknown, review",
            allowed_labels=("build", "review", "test"),
        )

        self.assertEqual(plan.decision_point, "need independent views")
        self.assertEqual(plan.selected_factors, ("review", "build"))
        self.assertFalse(plan.used_default_factors)

    def test_parse_branch_plan_falls_back_to_allowed_mentions(self) -> None:
        plan = parse_branch_plan(
            "I would fork for test and review now.",
            allowed_labels=("build", "review", "test"),
        )

        self.assertEqual(plan.selected_factors, ("test", "review"))
        self.assertFalse(plan.used_default_factors)

    def test_rewrite_creates_one_active_consideration_per_branch(self) -> None:
        plan = parse_branch_plan(
            "DECISION_POINT: before implementation\nFACTORS: build, review",
            allowed_labels=("build", "review", "test"),
        )

        stacked = stacked_marker_for_report(plan)
        markers = rewrite_stacked_factors(
            plan,
            considerations=DEFAULT_CONSIDERATIONS,
        )

        self.assertIn("Selected factors: build, review", stacked)
        self.assertEqual(set(markers), {"build", "review"})
        self.assertIn("implementation step", markers["build"])
        self.assertIn("STOP_POINT", markers["build"])
        self.assertNotIn("runtime failure mode", markers["build"])
        self.assertIn("runtime failure mode", markers["review"])
        self.assertNotIn("implementation step", markers["review"])

    def test_planner_prompt_names_bounded_catalog(self) -> None:
        prompt = render_planner_prompt(task="pick a fork")

        self.assertIn("Allowed factors:", prompt)
        self.assertIn("DECISION_POINT:", prompt)
        self.assertIn("FACTORS:", prompt)
        self.assertIn("build", prompt)
        self.assertIn("review", prompt)
        self.assertIn("test", prompt)

    def test_plan_records_default_factor_fallback(self) -> None:
        plan = parse_branch_plan(
            "DECISION_POINT: ambiguous\nFACTORS: none",
            allowed_labels=("build", "review", "test"),
        )

        self.assertTrue(plan.used_default_factors)


if __name__ == "__main__":
    unittest.main()
