from __future__ import annotations

import unittest

from localised_reasoning.ollama_branching import (
    BUILD_BRANCH,
    REVIEW_BRANCH,
    assess_branch_pair,
    assess_branch_output,
    build_branch_messages,
    build_branch_prompt,
    run_marker_bias_probe,
    strip_control_markers,
)


class OllamaBranchingTests(unittest.TestCase):
    def test_prompt_includes_marker_as_runtime_control(self) -> None:
        prompt = build_branch_prompt(
            shared_context="Shared prefix.",
            branch=REVIEW_BRANCH,
        )

        self.assertIn("TRUSTED_RUNTIME_CONTROL:", prompt)
        self.assertIn(REVIEW_BRANCH.marker, prompt)
        self.assertIn("Do not print them", prompt)
        self.assertIn("smallest test that would expose it", prompt)

    def test_messages_put_runtime_control_in_system_role(self) -> None:
        messages = build_branch_messages(
            shared_context="Shared prefix.",
            branch=BUILD_BRANCH,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "system")
        self.assertIn(BUILD_BRANCH.marker, messages[2]["content"])

    def test_assessment_accepts_good_builder_output(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks immediately after the shared prefix "
            "fork. The design decision is immutable shared prefix ownership with "
            "copy-on-append suffix writes after the hidden marker is ingested "
            "before visible continuation tokens, and visible tokens are conditioned "
            "by shared prefix plus branch-local suffix state. The verification target "
            "is a unit test that confirms prefix blocks are unchanged."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertTrue(assessment.passed, assessment.issues)

    def test_assessment_rejects_bad_mechanics(self) -> None:
        output = (
            "Inject at a fractional token offset and inspect hidden-state activation. "
            "This should work."
        )

        assessment = assess_branch_output(REVIEW_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertTrue(assessment.issues)

    def test_assessment_rejects_line_breaks(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork.\n"
            "The design decision is shared prefix ownership with branch-local "
            "suffix allocation. The verification target is cache metadata."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn("output contains line breaks", assessment.issues)

    def test_assessment_rejects_wrong_marker_bias_expectation(self) -> None:
        output = (
            "Compare the no-marker condition, inert-user-text condition, and "
            "trusted-runtime-control condition. The design decision is shared "
            "prefix ownership with branch-local suffix blocks. The verification "
            "target is that all three are statistically indistinguishable."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "incorrect marker-bias expectation across trusted control",
            assessment.issues,
        )

    def test_assessment_rejects_builder_missing_marker_ordering(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks immediately after the shared "
            "prefix fork. The design decision is immutable shared prefix ownership "
            "with branch-local suffix allocation. The verification target is cache "
            "metadata for forbidden runtime cache writes."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "builder missing hidden-marker-before-visible ordering",
            assessment.issues,
        )

    def test_assessment_rejects_builder_missing_conditioning(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks immediately after the shared "
            "prefix fork. The design decision is immutable shared prefix ownership "
            "after the hidden marker is ingested before visible continuation "
            "tokens. The verification target is cache metadata for forbidden "
            "runtime cache writes."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "builder missing shared-prefix-plus-suffix conditioning",
            assessment.issues,
        )

    def test_assessment_rejects_visible_transcript_append(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is shared prefix ownership with branch-local "
            "suffix allocation appended to the visible transcript. The verification "
            "target is a unit test for prefix immutability."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "incorrectly appends hidden branch state to visible transcript",
            assessment.issues,
        )

    def test_assessment_rejects_fabricated_marker_buffer(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is shared prefix ownership with a zero-filled "
            "token buffer for marker insertion. The verification target is a unit "
            "test for prefix immutability."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertTrue(any("zero" in issue for issue in assessment.issues))

    def test_assessment_rejects_prefix_disconnection(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is shared prefix ownership with branch-local "
            "suffix allocation. The verification target is that generated output "
            "does not depend on shared prefix content beyond the fork point."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertTrue(any("beyond the fork point" in issue for issue in assessment.issues))

    def test_assessment_rejects_attention_leakage_language(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is shared prefix ownership with no attention "
            "leakage. The verification target is a unit test for prefix immutability."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertTrue(any("attention leakage" in issue for issue in assessment.issues))

    def test_assessment_rejects_reference_mutation_conflation(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is shared prefix ownership with branch-local "
            "suffix allocation. The verification target is that continuation "
            "tokens do not reference the shared prefix."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertTrue(any("reference the shared prefix" in issue for issue in assessment.issues))

    def test_assessment_rejects_output_text_as_write_verification(self) -> None:
        output = (
            "The most likely failure mode is a write to shared prefix KV blocks. "
            "This matters because it can leak state across branches. The smallest "
            "exposing test is to verify the generated output does not contain any "
            "write to shared prefix KV blocks."
        )

        assessment = assess_branch_output(REVIEW_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "confuses generated text with cache write verification",
            assessment.issues,
        )

    def test_assessment_rejects_generated_tokens_as_write_actors(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is shared prefix ownership with branch-local "
            "suffix allocation. The verification target is that no generated "
            "token writes to shared prefix KV blocks."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "describes generated tokens as KV write actors",
            assessment.issues,
        )

    def test_assessment_rejects_generated_tokens_as_mutation_actors(self) -> None:
        output = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is shared prefix ownership with branch-local "
            "suffix allocation. The verification target is that no generated "
            "token in any branch modifies the shared prefix KV blocks."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "describes generated tokens as KV write actors",
            assessment.issues,
        )

    def test_assessment_rejects_kv_blocks_as_write_actors(self) -> None:
        output = (
            "The most likely failure mode is state leakage between branches. "
            "This matters because isolation can fail. The smallest exposing "
            "test is verifying that branch-local suffix KV blocks write to "
            "shared prefix KV blocks only through approved paths."
        )

        assessment = assess_branch_output(REVIEW_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "describes KV blocks as write actors",
            assessment.issues,
        )

    def test_assessment_rejects_marker_bias_without_inert_condition(self) -> None:
        output = (
            "The most likely failure mode is marker-specific drift in the "
            "trusted-runtime-control condition. This matters because branch "
            "instructions can bias outputs. The smallest exposing test is to "
            "compare generated outputs from the no-marker condition and verify "
            "cache metadata for shared-prefix writes."
        )

        assessment = assess_branch_output(REVIEW_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "marker-bias validation omitted inert-user-text condition",
            assessment.issues,
        )

    def test_assessment_rejects_exact_marker_bias_text_matching(self) -> None:
        output = (
            "Compare the no-marker condition and inert-user-text condition for "
            "semantic equivalence. The design decision is shared prefix ownership "
            "with branch-local suffix blocks. The verification target is that the "
            "outputs match exactly."
        )

        assessment = assess_branch_output(BUILD_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "requires exact marker-bias text match instead of semantic equivalence",
            assessment.issues,
        )

    def test_assessment_rejects_trusted_control_equivalence_requirement(self) -> None:
        output = (
            "The most likely failure mode is marker-specific drift in the "
            "trusted-runtime-control condition. This matters because branch "
            "instructions can bias outputs. The smallest exposing test is to "
            "compare the no-marker condition and inert-user-text condition, then "
            "verify that the trusted-runtime-control condition does not produce "
            "outputs that differ in content or reasoning from these."
        )

        assessment = assess_branch_output(REVIEW_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "incorrect marker-bias expectation across trusted control",
            assessment.issues,
        )

    def test_assessment_rejects_reviewer_without_cache_evidence_source(self) -> None:
        output = (
            "The most likely failure mode is shared prefix mutation during branch "
            "continuation. This matters because branch state can leak across "
            "reasoning paths. The smallest exposing test is to verify the trusted "
            "runtime behavior has no shared prefix mutations."
        )

        assessment = assess_branch_output(REVIEW_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "reviewer missing cache-metadata evidence source",
            assessment.issues,
        )

    def test_assessment_rejects_reviewer_without_exposing_test_opening(self) -> None:
        output = (
            "The most likely failure mode is shared prefix mutation during branch "
            "continuation. This matters because branch state can leak across "
            "reasoning paths. The immediate verification target is cache metadata, "
            "block ownership, refcounts, and write counters."
        )

        assessment = assess_branch_output(REVIEW_BRANCH, output)
        self.assertFalse(assessment.passed)
        self.assertIn(
            "reviewer third sentence does not start with exposing test",
            assessment.issues,
        )

    def test_pair_assessment_accepts_distinct_roles(self) -> None:
        builder = (
            "Allocate branch-local suffix KV blocks after the shared prefix fork. "
            "The design decision is immutable prefix ownership with copy-on-append "
            "suffix writes. The verification target is a unit test for unchanged "
            "prefix block refcounts."
        )
        reviewer = (
            "The most likely failure mode is write protection failure on shared "
            "prefix blocks. This matters because branch state can leak across "
            "continuations and violate isolation. The smallest exposing test is "
            "to mutate one suffix and verify all prefix ownership metadata stays "
            "unchanged."
        )

        assessment = assess_branch_pair(builder, reviewer)
        self.assertTrue(assessment.passed, assessment.issues)

    def test_pair_assessment_rejects_role_collapse(self) -> None:
        builder = (
            "The next engineering step is to validate prefix ownership. The design "
            "decision is shared prefix blocks with branch-local suffix blocks. The "
            "verification target is a unit test."
        )
        reviewer = builder

        assessment = assess_branch_pair(builder, reviewer)
        self.assertFalse(assessment.passed)
        self.assertTrue(assessment.issues)

    def test_probe_api_requires_runtime_model_call_parameters(self) -> None:
        self.assertTrue(callable(run_marker_bias_probe))

    def test_probe_assessment_rejects_diagnostic_carrier_leakage(self) -> None:
        from localised_reasoning.ollama_branching import _assess_marker_probe

        issues = _assess_marker_probe(
            branch=BUILD_BRANCH,
            no_marker_output="This mentions diagnostic text.",
            inert_user_text_output="Normal output.",
            trusted_runtime_control_output=(
                "Allocate branch-local suffix KV blocks after the shared prefix fork. "
                "Visible continuation tokens are conditioned by the shared prefix plus "
                "branch-local suffix state. The verification target is a unit test."
            ),
            inert_similarity=0.9,
            trusted_similarity=0.2,
        )

        self.assertTrue(any("diagnostic" in issue for issue in issues))

    def test_strip_control_markers_removes_reserved_text(self) -> None:
        output = (
            f"hello {BUILD_BRANCH.marker} world "
            f"{REVIEW_BRANCH.marker}"
        )

        self.assertEqual(strip_control_markers(output), "hello  world")


if __name__ == "__main__":
    unittest.main()
