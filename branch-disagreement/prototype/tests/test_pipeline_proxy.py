import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from branch_disagreement.clustering import ExactMatchClusterer  # noqa: E402
from branch_disagreement.config import ExperimentConfig  # noqa: E402
from branch_disagreement.model_runner import ProxyRunner  # noqa: E402
from branch_disagreement.pipeline import run_experiment  # noqa: E402


class TestProxyPipeline(unittest.TestCase):
    def _run(self, branch_mode="self_consistency", seed=0):
        config = ExperimentConfig(
            engine="proxy", dataset="sample", limit=12, n_branches=8,
            branch_mode=branch_mode, bootstrap_samples=200, seed=seed,
        )
        return run_experiment(config, ProxyRunner(seed=seed), ExactMatchClusterer())

    def test_runs_end_to_end(self):
        result = self._run()
        self.assertEqual(result.n_questions, 12)
        self.assertEqual(len(result.detectors), 5)
        self.assertTrue(result.total_generated_tokens > 0)

    def test_all_aucs_in_range(self):
        result = self._run()
        for d in result.detectors:
            self.assertGreaterEqual(d.auc, 0.0)
            self.assertLessEqual(d.auc, 1.0)
            self.assertLessEqual(d.ci_low, d.auc + 1e-9)
            self.assertGreaterEqual(d.ci_high, d.auc - 1e-9)

    def test_disagreement_beats_chance_on_proxy(self):
        # The proxy wires disagreement to track the latent "knows" state, so the
        # detector must clear chance. This is a process check, not a quality claim.
        result = self._run()
        branch = next(d for d in result.detectors if d.name == "branch_disagreement")
        self.assertGreater(branch.auc, 0.5)

    def test_delong_populated_for_baselines_only(self):
        result = self._run()
        for d in result.detectors:
            if d.name == "branch_disagreement":
                self.assertIsNone(d.delong_p_vs_branch)
            else:
                self.assertIsNotNone(d.delong_p_vs_branch)

    def test_localised_mode_runs(self):
        result = self._run(branch_mode="localised")
        self.assertEqual(result.n_questions, 12)
        # markers should be attached in localised mode
        self.assertTrue(result.total_generated_tokens > 0)

    def test_deterministic(self):
        a = self._run(seed=3)
        b = self._run(seed=3)
        self.assertEqual(
            [r.predicted for r in a.rows], [r.predicted for r in b.rows]
        )


if __name__ == "__main__":
    unittest.main()
