from __future__ import annotations

import unittest

from localised_reasoning.branch_evaluation import (
    branch_evaluation_prompt,
    validate_branch_evaluation_payload,
)


class BranchEvaluationTests(unittest.TestCase):
    def test_validate_branch_evaluation_payload_accepts_resolution(self) -> None:
        validate_branch_evaluation_payload(
            {
                "evaluations": [
                    {
                        "branch_id": "build",
                        "factor": "build",
                        "stop_point": "implementation complete",
                        "relevance": 4,
                        "novelty": 3,
                        "correctness_risk": 2,
                        "actionability": 5,
                        "decision": "keep",
                        "rationale": "Concrete next step.",
                    }
                ],
                "resolution": {
                    "decision": "merge",
                    "selected_branch_ids": ["build"],
                    "merged_summary": "Use the build step.",
                    "next_prompt": "",
                },
            }
        )

    def test_branch_evaluation_prompt_contains_branch_outputs(self) -> None:
        prompt = branch_evaluation_prompt(
            {
                "task": "advance runtime",
                "decision_point": "fork now",
                "branches": [
                    {
                        "branch_id": "review",
                        "factor": "review",
                        "stop_point": "risk identified",
                        "output": "cache corruption is likely",
                    }
                ],
            }
        )

        self.assertIn("advance runtime", prompt)
        self.assertIn("cache corruption is likely", prompt)
        self.assertIn("Resolution decisions", prompt)


if __name__ == "__main__":
    unittest.main()
