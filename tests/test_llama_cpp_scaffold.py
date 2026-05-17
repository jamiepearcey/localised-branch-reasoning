from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LlamaCppScaffoldTest(unittest.TestCase):
    def test_demo_uses_llama_sequence_memory_fork(self):
        source = (ROOT / "cpp" / "llama_seq_fork_demo.cpp").read_text()

        self.assertIn("llama_memory_seq_cp(mem, 0, 1", source)
        self.assertIn("llama_memory_seq_cp(mem, 0, 2", source)
        self.assertIn("prefix_forward_passes=1", source)
        self.assertIn("sequence_copy_api=llama_memory_seq_cp", source)

    def test_demo_keeps_marker_text_out_of_visible_sections(self):
        source = (ROOT / "cpp" / "llama_seq_fork_demo.cpp").read_text()

        self.assertIn("branch_a_hidden_marker_tokens", source)
        self.assertIn("branch_b_hidden_marker_tokens", source)
        self.assertIn("branch_a_visible_control_leaks", source)
        self.assertIn("branch_b_visible_control_leaks", source)
        self.assertIn("--output-file", source)
        self.assertNotIn("branch_a_marker ===", source)
        self.assertNotIn("branch_b_marker ===", source)

    def test_demo_includes_planned_factor_stop_mode(self):
        source = (ROOT / "cpp" / "llama_seq_fork_demo.cpp").read_text()

        self.assertIn("--planned", source)
        self.assertIn("DECISION_POINT:", source)
        self.assertIn("FACTORS:", source)
        self.assertIn("STOP_POINT:", source)
        self.assertIn("model_stop_detected", source)
        self.assertIn("planner_selected_factors", source)

    def test_build_script_links_homebrew_llama(self):
        script = (ROOT / "scripts" / "build_llama_seq_fork_demo.sh").read_text()

        self.assertIn("-I/opt/homebrew/include", script)
        self.assertIn("-L/opt/homebrew/lib", script)
        self.assertIn("-lllama", script)


if __name__ == "__main__":
    unittest.main()
