import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from branch_disagreement.normalize import answer_matches, normalize_answer  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_strips_articles_punct_case(self):
        self.assertEqual(normalize_answer("The Paris!"), "paris")
        self.assertEqual(normalize_answer("  a   CAT.  "), "cat")

    def test_none_and_empty(self):
        self.assertEqual(normalize_answer(None), "")
        self.assertEqual(normalize_answer(""), "")


class TestAnswerMatches(unittest.TestCase):
    def test_exact_after_normalization(self):
        self.assertTrue(answer_matches("Paris.", ["paris"]))
        self.assertTrue(answer_matches("the George Washington", ["George Washington"]))

    def test_span_match_inside_longer_answer(self):
        self.assertTrue(
            answer_matches("The answer is George Washington, I think", ["George Washington"])
        )

    def test_no_false_positive_on_substring_token(self):
        # "art" should not match inside "Washington"-style tokens; token-span only
        self.assertFalse(answer_matches("Washington", ["ash"]))

    def test_wrong_answer(self):
        self.assertFalse(answer_matches("Lyon", ["Paris"]))

    def test_empty_prediction(self):
        self.assertFalse(answer_matches("", ["Paris"]))

    def test_multiple_golds(self):
        self.assertTrue(answer_matches("Shakespeare", ["William Shakespeare", "Shakespeare"]))


if __name__ == "__main__":
    unittest.main()
