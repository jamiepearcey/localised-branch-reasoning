import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from localised_reasoning.swebench_eval import (
    ProxySWEBenchBranchEngine,
    SWEBenchInstance,
    build_swebench_branch_eval,
    build_swebench_coding_cases,
    export_swebench_branch_eval,
    parse_patch_file_paths,
    render_swebench_final_patch_prompt,
    write_swebench_outputs,
)


def sample_instance() -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id="example__repo-1",
        dataset="lite",
        split="test",
        repo="example/repo",
        base_commit="abc123",
        problem_statement="Function returns the wrong value for nested inputs.",
        hints_text="Focus on separability.",
        version="1.0",
        difficulty="",
        patch_file_paths=("pkg/module.py",),
        fail_to_pass='["pkg/tests/test_module.py::test_nested"]',
        pass_to_pass='["pkg/tests/test_module.py::test_existing"]',
    )


class SWEBenchEvalTest(unittest.TestCase):
    def test_parse_patch_file_paths(self):
        paths = parse_patch_file_paths(
            "diff --git a/pkg/a.py b/pkg/a.py\n"
            "--- a/pkg/a.py\n"
            "+++ b/pkg/a.py\n"
            "diff --git a/pkg/b.py b/pkg/b.py\n"
        )

        self.assertEqual(paths, ("pkg/a.py", "pkg/b.py"))

    def test_build_case_excludes_gold_patch_and_uses_source_context(self):
        instance = sample_instance()
        cases, context_df = build_swebench_coding_cases(
            [instance],
            source_mode="none",
        )

        self.assertEqual(len(cases), 1)
        self.assertIn("Function returns the wrong value", cases[0].code)
        self.assertIn("oracle_file_paths_for_context_only", cases[0].code)
        self.assertNotIn("diff --git", cases[0].code)
        self.assertEqual(context_df.iloc[0]["context_file_count"], 0)

    @patch("localised_reasoning.swebench_eval.fetch_base_file", return_value="def target():\n    return False\n")
    def test_build_case_fetches_base_source(self, _fetch):
        cases, context_df = build_swebench_coding_cases([sample_instance()])

        self.assertIn("L1: def target", cases[0].code)
        self.assertIn("0001: def target", cases[0].code)
        self.assertEqual(context_df.iloc[0]["context_file_paths"], "pkg/module.py")

    def test_patch_prompt_requests_only_unified_diff(self):
        case = build_swebench_coding_cases([sample_instance()], source_mode="none")[0][0]
        prompt = render_swebench_final_patch_prompt(case, {"method_start": "SUMMARY: bug is narrow"})

        self.assertIn("Output only a patch beginning with diff --git", prompt)
        self.assertIn("CHECKPOINT_COLLAPSE: method_start", prompt)

    def test_proxy_builds_workbook_and_predictions(self):
        (
            instance_df,
            case_df,
            branch_df,
            collapse_df,
            final_df,
            monolithic_df,
            patch_df,
            summary_df,
        ) = build_swebench_branch_eval(
            engine=ProxySWEBenchBranchEngine(),
            instances=[sample_instance()],
            source_mode="none",
            consideration_limit=2,
        )

        self.assertEqual(len(instance_df), 1)
        self.assertEqual(len(final_df), 1)
        self.assertTrue(patch_df["patch_valid_shape"].all())
        self.assertIn("branch_patch_valid_rate", set(summary_df["metric"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            branch_predictions, monolithic_predictions = write_swebench_outputs(
                output_dir=tmp,
                instance_df=instance_df,
                case_df=case_df,
                branch_df=branch_df,
                collapse_df=collapse_df,
                final_df=final_df,
                monolithic_df=monolithic_df,
                patch_df=patch_df,
                summary_df=summary_df,
                model_name="proxy",
            )
            output = export_swebench_branch_eval(
                output_xlsx=tmp / "swebench.xlsx",
                instance_df=instance_df,
                case_df=case_df,
                branch_df=branch_df,
                collapse_df=collapse_df,
                final_df=final_df,
                monolithic_df=monolithic_df,
                patch_df=patch_df,
                summary_df=summary_df,
            )
            self.assertTrue(branch_predictions.exists())
            self.assertTrue(monolithic_predictions.exists())
            self.assertTrue(output.exists())
            self.assertIn('"model_patch"', branch_predictions.read_text())


if __name__ == "__main__":
    unittest.main()
