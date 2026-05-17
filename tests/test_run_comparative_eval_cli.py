import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_comparative_eval.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_comparative_eval", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunComparativeEvalCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner_module()

    def test_resolves_default_model_preset(self):
        args = argparse.Namespace(model_path=None, model_preset="qwen3-30b-a3b")

        self.assertEqual(
            self.runner.resolve_model_path(args),
            Path("models/qwen3-30b-a3b-q4_k_m/Qwen3-30B-A3B-Q4_K_M.gguf"),
        )

    def test_resolves_small_model_preset(self):
        args = argparse.Namespace(model_path=None, model_preset="qwen3-4b")

        self.assertEqual(
            self.runner.resolve_model_path(args),
            Path("models/qwen3-4b-instruct-2507-q4_k_m/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        )

    def test_explicit_model_path_overrides_preset(self):
        args = argparse.Namespace(model_path=Path("models/custom.gguf"), model_preset="qwen3-4b")

        self.assertEqual(self.runner.resolve_model_path(args), Path("models/custom.gguf"))

    def test_missing_preset_model_error_includes_download_hint(self):
        with self.assertRaises(SystemExit) as error:
            self.runner.require_model_path(Path("missing.gguf"), "qwen3-14b")

        self.assertIn("huggingface-cli download unsloth/Qwen3-14B-GGUF", str(error.exception))


if __name__ == "__main__":
    unittest.main()
