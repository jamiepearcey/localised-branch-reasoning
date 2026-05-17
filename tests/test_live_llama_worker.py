from pathlib import Path
import unittest

from localised_reasoning.comparative_eval import EvalQuestion, LiveLlamaComparativeEngine
from localised_reasoning.qa_scenarios import ReasoningBudget, default_scenarios


ROOT = Path(__file__).resolve().parents[1]


class FakeWorker:
    def __init__(self) -> None:
        self.branch_requests = []
        self.generate_requests = []
        self.closed = False

    def branch(self, **kwargs):
        self.branch_requests.append(kwargs)
        return {
            "ok": True,
            "cmd": "branch",
            "parallel_decode": True,
            "prefix_forward_passes": 1,
            "sequence_copy_api": "llama_memory_seq_cp",
            "branches": [
                {"label": "fast_intuition", "text": "ANSWER: quick\nCONFIDENCE: 40\nEVIDENCE: first pass\nSTOP_POINT: complete"},
                {"label": "literal_check", "text": "ANSWER: literal\nCONFIDENCE: 80\nEVIDENCE: exact wording\nSTOP_POINT: complete"},
            ],
        }

    def generate(self, prompt, **kwargs):
        self.generate_requests.append((prompt, kwargs))
        if "Reasoned branch outputs" in prompt:
            text = "FINAL_ANSWER: literal\nCONFIDENCE: 72\nSELECTED_BRANCH: literal_check\nRATIONALE: literal branch is better"
        elif "conservative answer gate" in prompt:
            text = "FINAL_ANSWER: literal\nSOURCE: branch_selector\nCONFIDENCE: 74\nRATIONALE: branch evidence is specific"
        else:
            text = "ANSWER: literal\nCONFIDENCE: 70\nSUMMARY: checked wording"
        return {"ok": True, "cmd": "generate", "text": text}

    def close(self):
        self.closed = True


class CachedFakeWorker(FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.cache_requests = []
        self.cached_branch_requests = []
        self.cached_generate_requests = []

    def cache_prefix(self, **kwargs):
        self.cache_requests.append(kwargs)
        return {"ok": True, "cmd": "cache_prefix", "prefix_tokens": 12}

    def cached_branch(self, **kwargs):
        self.cached_branch_requests.append(kwargs)
        return self.branch(**kwargs)

    def cached_generate(self, **kwargs):
        self.cached_generate_requests.append(kwargs)
        return self.generate(kwargs["suffix"], **kwargs)


class LiveLlamaWorkerScaffoldTest(unittest.TestCase):
    def test_worker_cpp_uses_resident_model_and_sequence_copy(self):
        source = (ROOT / "cpp" / "llama_branch_worker.cpp").read_text()

        self.assertIn("llama_model_load_from_file", source)
        self.assertIn("while (std::getline(std::cin, line))", source)
        self.assertIn("cmd == \"generate\"", source)
        self.assertIn("cmd == \"branch\"", source)
        self.assertIn("cmd == \"cache_prefix\"", source)
        self.assertIn("cmd == \"cached_branch\"", source)
        self.assertIn("cmd == \"cached_generate\"", source)
        self.assertIn("llama_memory_seq_cp(mem, base_seq_id, seq_id", source)
        self.assertIn("parallel_decode", source)
        self.assertIn("prefix_forward_passes", source)

    def test_worker_build_script_links_homebrew_llama(self):
        script = (ROOT / "scripts" / "build_llama_branch_worker.sh").read_text()

        self.assertIn("llama_branch_worker.cpp", script)
        self.assertIn("-I/opt/homebrew/include", script)
        self.assertIn("-L/opt/homebrew/lib", script)
        self.assertIn("-lllama", script)

    def test_comparative_engine_uses_one_live_worker(self):
        worker = FakeWorker()
        engine = LiveLlamaComparativeEngine(model_path=Path("unused.gguf"), worker=worker)
        question = EvalQuestion(
            "t1",
            "test",
            "Which answer is literal?",
            "literal",
            ("literal",),
        )

        scenarios = default_scenarios()[:2]
        answers = engine.branch_answers(question, scenarios)
        judge = engine.adjudicate(question, answers)
        reasoning = engine.reason_answer(question, ReasoningBudget(2, 16, 8))
        engine.close()

        self.assertTrue(answers["fast_intuition"].startswith("quick"))
        self.assertTrue(answers["literal_check"].startswith("literal"))
        self.assertEqual(judge.final_answer, "literal")
        self.assertEqual(reasoning.answer, "literal")
        gate = engine.gate_answer(question, answers, judge, reasoning)
        self.assertEqual(gate.final_answer, "literal")
        self.assertEqual(gate.source, "branch_selector")
        self.assertEqual(len(worker.branch_requests), 1)
        self.assertEqual(len(worker.generate_requests), 3)
        self.assertTrue(worker.branch_requests[0]["parallel"])
        self.assertTrue(worker.closed)

    def test_comparative_engine_uses_cached_prefixes_when_available(self):
        worker = CachedFakeWorker()
        engine = LiveLlamaComparativeEngine(model_path=Path("unused.gguf"), worker=worker)
        question = EvalQuestion(
            "t1",
            "test",
            "Which answer is literal?",
            "literal",
            ("literal",),
        )

        scenarios = default_scenarios()[:2]
        answers = engine.branch_answers(question, scenarios)
        judge = engine.adjudicate(question, answers)
        reasoning = engine.reason_answer(question, ReasoningBudget(2, 16, 8))
        gate = engine.gate_answer(question, answers, judge, reasoning)

        self.assertTrue(engine.prefix_cache_enabled)
        self.assertEqual(len(worker.cache_requests), 3)
        self.assertEqual(len(worker.cached_branch_requests), 1)
        self.assertEqual(len(worker.cached_generate_requests), 2)
        self.assertEqual(worker.cached_branch_requests[0]["prefix_id"], "branch_question_prefix_v1")
        self.assertEqual(worker.cached_generate_requests[0]["prefix_id"], "judge_selector_prefix_v1")
        self.assertEqual(worker.cached_generate_requests[1]["prefix_id"], "meta_gate_prefix_v1")
        self.assertEqual(gate.final_answer, "literal")


if __name__ == "__main__":
    unittest.main()
