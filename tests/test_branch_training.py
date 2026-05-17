from __future__ import annotations

import unittest

from localised_reasoning.branch_training import (
    BranchingTrainingRecord,
    TrainingBranch,
    format_branching_sft_text,
    parse_planned_branch_report,
)


class BranchTrainingTests(unittest.TestCase):
    def test_format_branching_sft_text_contains_internalized_markers(self) -> None:
        text = format_branching_sft_text(
            BranchingTrainingRecord(
                task="advance runtime",
                decision_point="fork before implementation",
                factors=("build", "review"),
                branches=(
                    TrainingBranch(
                        branch_id="build",
                        factor="build",
                        output="Implement sequence copy.",
                        stop_point="step chosen",
                    ),
                ),
            )
        )

        self.assertIn("<LR_DECISION_POINT>", text)
        self.assertIn("<LR_FACTORS>build, review</LR_FACTORS>", text)
        self.assertIn("STOP_POINT: step chosen", text)

    def test_parse_planned_branch_report_extracts_training_record(self) -> None:
        report = """\
=== raw_planner_output ===
DECISION_POINT: fork here
FACTORS: build, review
=== stacked_marker_before_rewrite ===
ignored
=== rewritten_shared_prefix ===
Shared task:
advance runtime

Planner decision point:
fork here
=== rewritten_branch_markers ===
ignored
=== kv_fork_result ===
## branch=build
model_stop_point='done'
visible_text='Implement copy. STOP_POINT: done'
"""

        record = parse_planned_branch_report(report)

        self.assertEqual(record.task, "advance runtime")
        self.assertEqual(record.decision_point, "fork here")
        self.assertEqual(record.factors, ("build", "review"))
        self.assertEqual(record.branches[0].output, "Implement copy.")


if __name__ == "__main__":
    unittest.main()
