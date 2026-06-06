import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from branch_disagreement.clustering import ExactMatchClusterer  # noqa: E402
from branch_disagreement.model_runner import BranchSample  # noqa: E402
from branch_disagreement.scoring import (  # noqa: E402
    branch_disagreement,
    compute_scores,
    lexical_disagreement,
    neg_mean_logprob,
    primary_answer,
    semantic_entropy,
)


def make_branches(answers, logprob=-0.5):
    return [
        BranchSample(branch_id=i, text=a, answer=a, token_logprobs=[logprob, logprob],
                     num_tokens=2)
        for i, a in enumerate(answers)
    ]


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.clusterer = ExactMatchClusterer()

    def test_total_agreement_is_zero_disagreement(self):
        branches = make_branches(["Paris"] * 6)
        clusters = self.clusterer.cluster([b.answer for b in branches])
        self.assertEqual(branch_disagreement(branches, clusters), 0.0)
        self.assertEqual(lexical_disagreement(branches), 0.0)
        self.assertAlmostEqual(semantic_entropy(branches, clusters), 0.0)

    def test_total_disagreement_is_high(self):
        branches = make_branches(["a", "b", "c", "d"])
        clusters = self.clusterer.cluster([b.answer for b in branches])
        self.assertAlmostEqual(branch_disagreement(branches, clusters), 0.75)
        # four equiprobable clusters -> entropy = ln(4)
        self.assertAlmostEqual(semantic_entropy(branches, clusters), math.log(4))

    def test_partial_split(self):
        branches = make_branches(["Paris", "Paris", "Paris", "Lyon"])
        clusters = self.clusterer.cluster([b.answer for b in branches])
        self.assertAlmostEqual(branch_disagreement(branches, clusters), 0.25)
        self.assertAlmostEqual(lexical_disagreement(branches), 0.25)

    def test_primary_answer_is_majority(self):
        branches = make_branches(["Paris", "Paris", "Lyon"])
        clusters = self.clusterer.cluster([b.answer for b in branches])
        self.assertEqual(primary_answer(branches, clusters), "Paris")

    def test_neg_mean_logprob_sign(self):
        confident = make_branches(["Paris"] * 3, logprob=-0.1)
        unsure = make_branches(["Paris"] * 3, logprob=-2.0)
        # less confident => higher error score
        self.assertGreater(neg_mean_logprob(unsure), neg_mean_logprob(confident))

    def test_compute_scores_keys(self):
        branches = make_branches(["x", "y"])
        clusters = self.clusterer.cluster([b.answer for b in branches])
        scores = compute_scores(branches, clusters)
        self.assertEqual(
            set(scores),
            {"branch_disagreement", "semantic_entropy", "lexical_disagreement",
             "neg_mean_logprob", "neg_logprob_single"},
        )
        self.assertTrue(all(math.isfinite(v) for v in scores.values()))

    def test_likelihood_weighting_runs(self):
        branches = make_branches(["a", "a", "b"])
        clusters = self.clusterer.cluster([b.answer for b in branches])
        val = semantic_entropy(branches, clusters, weighting="likelihood")
        self.assertGreaterEqual(val, 0.0)


if __name__ == "__main__":
    unittest.main()
