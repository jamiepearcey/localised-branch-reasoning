import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from branch_disagreement.metrics import (  # noqa: E402
    bootstrap_auc_ci,
    delong_test,
    roc_auc,
)


class TestRocAuc(unittest.TestCase):
    def test_perfect_separation(self):
        # positives (error=1) all score higher than negatives
        scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
        labels = [0, 0, 0, 1, 1, 1]
        self.assertAlmostEqual(roc_auc(scores, labels), 1.0)

    def test_perfect_inversion(self):
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0]
        labels = [0, 0, 0, 1, 1, 1]
        self.assertAlmostEqual(roc_auc(scores, labels), 0.0)

    def test_chance_with_ties(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 1, 0, 1]
        self.assertAlmostEqual(roc_auc(scores, labels), 0.5)

    def test_known_value(self):
        # one positive ranked 2nd of 4 -> AUC = 0.5
        scores = [0.1, 0.3, 0.2, 0.05]
        labels = [0, 1, 0, 0]
        # positive score 0.3 beats 0.1,0.2,0.05 -> all 3 -> AUC=1.0
        self.assertAlmostEqual(roc_auc(scores, labels), 1.0)

    def test_empty_class_is_chance(self):
        self.assertEqual(roc_auc([0.1, 0.2], [0, 0]), 0.5)

    def test_half(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        labels = [1, 0, 1, 0]
        # positives {1.0,3.0}, negatives {2.0,4.0}: pairs won = (1>2?N)(1>4?N)(3>2?Y)(3>4?N)=1/4
        self.assertAlmostEqual(roc_auc(scores, labels), 0.25)


class TestBootstrap(unittest.TestCase):
    def test_ci_brackets_auc_and_is_deterministic(self):
        scores = [0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.85, 0.95]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        r1 = bootstrap_auc_ci(scores, labels, n_boot=500, seed=7)
        r2 = bootstrap_auc_ci(scores, labels, n_boot=500, seed=7)
        self.assertEqual((r1.ci_low, r1.ci_high), (r2.ci_low, r2.ci_high))
        self.assertLessEqual(r1.ci_low, r1.auc)
        self.assertGreaterEqual(r1.ci_high, r1.auc)


class TestDeLong(unittest.TestCase):
    def test_identical_predictors_not_significant(self):
        scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.9]
        labels = [0, 0, 1, 1, 0, 1]
        r = delong_test(scores, scores, labels)
        self.assertAlmostEqual(r.diff, 0.0)
        self.assertAlmostEqual(r.p_value, 1.0)
        self.assertAlmostEqual(r.auc_a, r.auc_b)

    def test_auc_matches_roc_auc(self):
        a = [0.2, 0.1, 0.6, 0.55, 0.3, 0.9, 0.8, 0.4]
        b = [0.5, 0.5, 0.4, 0.6, 0.5, 0.7, 0.3, 0.5]
        labels = [0, 0, 1, 1, 0, 1, 1, 0]
        r = delong_test(a, b, labels)
        self.assertAlmostEqual(r.auc_a, roc_auc(a, labels), places=9)
        self.assertAlmostEqual(r.auc_b, roc_auc(b, labels), places=9)

    def test_strong_vs_weak_detector_direction(self):
        # a separates classes well, b is near chance; z should be positive
        a = [0.1, 0.15, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95]
        b = [0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.53, 0.47]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        r = delong_test(a, b, labels)
        self.assertGreater(r.auc_a, r.auc_b)
        self.assertGreater(r.z, 0.0)
        self.assertTrue(math.isfinite(r.p_value))
        self.assertGreaterEqual(r.p_value, 0.0)
        self.assertLessEqual(r.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
