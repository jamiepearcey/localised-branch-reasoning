import unittest
from unittest import mock

from localised_reasoning.benchmark_datasets import (
    _mmlu_pro_row_to_question,
    load_mmlu_pro_questions,
)
from localised_reasoning.comparative_eval import ComparativeProxyEngine, score_answer
from localised_reasoning.qa_scenarios import ReasoningBudget, default_scenarios


SAMPLE_ROWS = [
    {
        "question_id": 101,
        "question": "Which number is prime?",
        "options": ["4", "6", "7", "8", "9", "10", "12", "14", "15", "21"],
        "answer": "C",
        "answer_index": 2,
        "cot_content": "hidden",
        "category": "math",
        "src": "unit-test",
    },
    {
        "question_id": 102,
        "question": "Which keyword defines a Python function?",
        "options": ["class", "def", "return", "yield", "with", "try", "for", "if", "else", "lambda"],
        "answer": "B",
        "answer_index": 1,
        "cot_content": "hidden",
        "category": "computer science",
        "src": "unit-test",
    },
]


class BenchmarkDatasetTest(unittest.TestCase):
    def test_mmlu_pro_row_formats_multiple_choice_question(self):
        question = _mmlu_pro_row_to_question(SAMPLE_ROWS[0])

        self.assertEqual(question.question_id, "mmlu-pro-101")
        self.assertIn("Options:", question.question)
        self.assertIn("C. 7", question.question)
        self.assertEqual(question.expected_answer, "C. 7")
        self.assertTrue(score_answer("C", question))
        self.assertTrue(score_answer("The answer is C.", question))
        self.assertTrue(score_answer("7", question))
        self.assertFalse(score_answer("B", question))

    def test_load_mmlu_pro_filters_categories_and_limits(self):
        with mock.patch(
            "localised_reasoning.benchmark_datasets._load_mmlu_pro_local_rows",
            return_value=None,
        ), mock.patch(
            "localised_reasoning.benchmark_datasets._mmlu_pro_total_rows",
            return_value=2,
        ), mock.patch(
            "localised_reasoning.benchmark_datasets._fetch_mmlu_pro_rows",
            return_value=SAMPLE_ROWS,
        ):
            questions = load_mmlu_pro_questions(
                limit=1,
                categories=["computer science"],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].question_id, "mmlu-pro-102")

    def test_proxy_can_smoke_test_generic_benchmark_questions(self):
        question = _mmlu_pro_row_to_question(SAMPLE_ROWS[0])
        engine = ComparativeProxyEngine()
        answers = engine.branch_answers(question, default_scenarios())
        judgement = engine.adjudicate(question, answers)
        reasoning = engine.reason_answer(question, ReasoningBudget(5, 160, 48))

        self.assertTrue(score_answer(judgement.final_answer, question))
        self.assertTrue(score_answer(reasoning.answer, question))


if __name__ == "__main__":
    unittest.main()
