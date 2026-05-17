import tempfile
from pathlib import Path
import unittest

from localised_reasoning.long_decision_eval import (
    ProxyLongDecisionEngine,
    build_long_decision_eval,
    default_long_decision_cases,
    export_long_decision_eval,
    parse_branch_plan,
    parse_factor_blocks,
    parse_final_decision_score,
    render_branch_planner_prompt,
    render_factor_branch_marker,
    render_final_decision_judge_prompt,
    render_forked_aggregation_prompt,
    render_sequential_step_prompt,
    score_factor_output,
)


class LongDecisionEvalTest(unittest.TestCase):
    def test_planner_prompt_uses_catalog_without_expected_answer(self):
        case = default_long_decision_cases()[0]
        prompt = render_branch_planner_prompt(case)

        self.assertIn("Allowed factors:", prompt)
        self.assertIn("BRANCH_DECISION:", prompt)
        self.assertIn("latency", prompt)
        self.assertIn(case.context, prompt)

    def test_parse_branch_plan_uses_allowed_labels(self):
        case = default_long_decision_cases()[0]
        plan = parse_branch_plan(
            "BRANCH_DECISION: yes\nDECISION_POINT: separate constraints\nFACTORS: latency, compliance, made_up, cost",
            case,
        )

        self.assertTrue(plan.should_branch)
        self.assertEqual(plan.factors, ("latency", "compliance", "cost"))

    def test_factor_marker_is_local_and_has_stop_point(self):
        factor = default_long_decision_cases()[0].factors[0]
        marker = render_factor_branch_marker(factor)

        self.assertIn("Active local factor: latency", marker)
        self.assertIn("Evaluate only this factor", marker)
        self.assertIn("STOP_POINT:", marker)

    def test_sequential_step_prompt_carries_prior_outputs(self):
        case = default_long_decision_cases()[0]
        prompt = render_sequential_step_prompt(
            case,
            case.factors[:2],
            case.factors[1],
            ["FACTOR: latency\nLOCAL_RECOMMENDATION: prior latency conclusion"],
        )

        self.assertIn("Previous factor outputs:", prompt)
        self.assertIn("prior latency conclusion", prompt)
        self.assertIn("Active factor now: compliance", prompt)

    def test_case_suffix_does_not_stack_selected_factors_into_branches(self):
        from localised_reasoning.long_decision_eval import BranchPlan, render_case_suffix

        case = default_long_decision_cases()[0]
        suffix = render_case_suffix(
            case,
            BranchPlan(True, "separable factors", ("latency", "compliance", "cost"), "raw"),
        )

        self.assertIn(case.context, suffix)
        self.assertNotIn("Selected localized factors", suffix)
        self.assertNotIn("latency, compliance, cost", suffix)

    def test_aggregation_prompt_collapses_branch_artifacts(self):
        case = default_long_decision_cases()[0]
        prompt = render_forked_aggregation_prompt(
            case,
            case.factors[:2],
            {
                "latency": "FACTOR: latency\nLOCAL_RECOMMENDATION: split read model",
                "compliance": "FACTOR: compliance\nLOCAL_RECOMMENDATION: audit isolation",
            },
        )

        self.assertIn("Artifact source: independent localized factor artifacts", prompt)
        self.assertIn("Factor artifacts:", prompt)
        self.assertIn("FINAL_DECISION:", prompt)
        self.assertIn("split read model", prompt)

    def test_final_decision_judge_prompt_and_parser(self):
        case = default_long_decision_cases()[0]
        prompt = render_final_decision_judge_prompt(
            case,
            case.factors[:2],
            "forked_aggregate",
            "FINAL_DECISION: staged hybrid",
        )
        self.assertIn("GROUNDEDNESS:", prompt)
        self.assertIn("Candidate method: forked_aggregate", prompt)

        score = parse_final_decision_score(
            "GROUNDEDNESS: 4\n"
            "FACTOR_COVERAGE: 5\n"
            "SYNTHESIS_QUALITY: 3\n"
            "RISK_HANDLING: 4\n"
            "ACTIONABILITY: 5\n"
            "OVERALL: 4\n"
            "RATIONALE: solid"
        )
        self.assertEqual(score.factor_coverage, 5)
        self.assertEqual(score.overall, 4)
        self.assertEqual(score.rationale, "solid")

    def test_score_detects_contamination(self):
        case = default_long_decision_cases()[0]
        clean = score_factor_output(case, "latency", "Dashboard latency must hit 1.5 seconds with a read model.")
        contaminated = score_factor_output(case, "latency", "Latency matters, but CFO spend and PHI audit isolation dominate.")

        self.assertGreater(clean.focus_score, contaminated.focus_score)
        self.assertGreater(contaminated.contamination_count, 0)

    def test_score_ignores_tradeoff_boundary_for_contamination(self):
        case = default_long_decision_cases()[0]
        score = score_factor_output(
            case,
            "latency",
            (
                "FACTOR: latency\n"
                "LOCAL_RECOMMENDATION: Use a read model for dashboard latency.\n"
                "EVIDENCE: Dashboards must load within 1.5 seconds.\n"
                "TRADEOFF_BOUNDARY: This does not decide PHI compliance, CFO spend, or rollback migration risk.\n"
            ),
        )

        self.assertEqual(score.contamination_count, 0)
        self.assertEqual(score.focus_score, 5)

    def test_parse_factor_blocks(self):
        blocks = parse_factor_blocks(
            "FACTOR: latency\nLOCAL_RECOMMENDATION: x\n\nFACTOR: cost\nLOCAL_RECOMMENDATION: y",
            ["latency", "cost"],
        )

        self.assertIn("LOCAL_RECOMMENDATION: x", blocks["latency"])
        self.assertIn("LOCAL_RECOMMENDATION: y", blocks["cost"])

    def test_proxy_builds_exportable_eval(self):
        case_df, branch_df, sequential_df, monolithic_df, aggregate_df, judge_df, summary_df = build_long_decision_eval(
            engine=ProxyLongDecisionEngine(),
            cases=default_long_decision_cases()[:1],
        )

        self.assertEqual(len(case_df), 1)
        self.assertGreaterEqual(len(branch_df), 3)
        self.assertEqual(len(branch_df), len(sequential_df))
        self.assertEqual(len(aggregate_df), 2)
        self.assertEqual(len(judge_df), 3)
        self.assertIn("forked_branch_avg_focus_score", set(summary_df["metric"]))
        self.assertIn("avg_forked_aggregate_factor_coverage", set(summary_df["metric"]))
        self.assertIn("forked_aggregate_judge_overall", set(summary_df["metric"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_long_decision_eval(
                output_xlsx=Path(tmpdir) / "long_decision.xlsx",
                case_df=case_df,
                branch_df=branch_df,
                sequential_df=sequential_df,
                monolithic_df=monolithic_df,
                aggregate_df=aggregate_df,
                judge_df=judge_df,
                summary_df=summary_df,
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
