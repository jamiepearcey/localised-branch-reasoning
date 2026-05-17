import tempfile
from pathlib import Path
import unittest

from localised_reasoning.coding_branch_eval import (
    ProxyCodingBranchEngine,
    build_coding_branch_eval,
    default_coding_cases,
    export_coding_branch_eval,
    parse_coding_branch_plan,
    render_checkpoint_collapse_prompt,
    render_coding_consideration_marker,
    render_coding_planner_prompt,
    score_coding_branch_points,
)


class CodingBranchEvalTest(unittest.TestCase):
    def test_planner_prompt_prioritizes_high_risk_method_points(self):
        case = default_coding_cases()[0]
        prompt = render_coding_planner_prompt(case)

        self.assertIn("method start", prompt)
        self.assertIn("method end", prompt)
        self.assertIn("external side effects", prompt)
        self.assertIn("BRANCH_POINTS:", prompt)

    def test_parse_plan_keeps_allowed_labels(self):
        case = default_coding_cases()[0]
        plan = parse_coding_branch_plan(
            "BRANCH_POINTS: method_start, made_up, external_call, method_end\nRATIONALE: risky boundaries",
            case,
        )

        self.assertEqual(plan.selected_points, ("method_start", "external_call", "method_end"))
        score = score_coding_branch_points(case, plan)
        self.assertTrue(score.includes_method_start)
        self.assertTrue(score.includes_method_end)

    def test_consideration_marker_is_checkpoint_local(self):
        case = default_coding_cases()[0]
        marker = render_coding_consideration_marker(case.branch_points[0], case.considerations[0])

        self.assertIn("Active checkpoint: method_start", marker)
        self.assertIn("Active consideration: contract", marker)
        self.assertIn("STOP_POINT:", marker)

    def test_checkpoint_collapse_uses_branch_artifacts(self):
        case = default_coding_cases()[0]
        prompt = render_checkpoint_collapse_prompt(
            case,
            case.branch_points[0],
            {"contract": "FINDING: validate amount", "tests": "FINDING: test forbidden actor"},
        )

        self.assertIn("Collapse local coding branch artifacts", prompt)
        self.assertIn("BRANCH_ARTIFACT: contract", prompt)
        self.assertIn("CHECKPOINT:", prompt)

    def test_proxy_builds_multiple_collapses(self):
        case_df, branch_df, collapse_df, final_df, monolithic_df, summary_df = build_coding_branch_eval(
            engine=ProxyCodingBranchEngine(),
            cases=default_coding_cases()[:1],
            consideration_limit=3,
        )

        self.assertEqual(len(case_df), 1)
        self.assertGreaterEqual(len(collapse_df), 3)
        self.assertEqual(len(final_df), 1)
        self.assertEqual(len(monolithic_df), 1)
        self.assertIn("method_start_rate", set(summary_df["metric"]))
        self.assertIn("collapse_count", set(summary_df["metric"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_coding_branch_eval(
                output_xlsx=Path(tmpdir) / "coding.xlsx",
                case_df=case_df,
                branch_df=branch_df,
                collapse_df=collapse_df,
                final_df=final_df,
                monolithic_df=monolithic_df,
                summary_df=summary_df,
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
