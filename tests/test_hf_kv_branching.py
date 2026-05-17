from __future__ import annotations

import unittest

import torch
from transformers.cache_utils import DynamicCache

from localised_reasoning.hf_kv_branching import (
    cache_seq_length,
    caches_share_tensor_storage,
    clone_cache_tensors_to_cpu,
    cache_tensors_equal,
    extract_model_stop_point,
    shallow_fork_dynamic_cache,
    visible_control_leaks_for,
)


class HFKVBranchingTests(unittest.TestCase):
    def test_shallow_fork_shares_prefix_storage_until_append(self) -> None:
        cache = DynamicCache()
        key = torch.randn(1, 2, 3, 4)
        value = torch.randn(1, 2, 3, 4)
        cache.update(key, value, layer_idx=0)

        fork = shallow_fork_dynamic_cache(cache)

        self.assertTrue(caches_share_tensor_storage(cache, fork))
        self.assertEqual(cache_seq_length(fork), 3)

    def test_fork_append_does_not_mutate_prefix_cache(self) -> None:
        cache = DynamicCache()
        key = torch.randn(1, 2, 3, 4)
        value = torch.randn(1, 2, 3, 4)
        cache.update(key, value, layer_idx=0)
        prefix_snapshot = clone_cache_tensors_to_cpu(cache)

        fork = shallow_fork_dynamic_cache(cache)
        fork.update(torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4), layer_idx=0)

        self.assertTrue(cache_tensors_equal(cache, prefix_snapshot))
        self.assertEqual(cache_seq_length(cache), 3)
        self.assertEqual(cache_seq_length(fork), 5)
        self.assertFalse(caches_share_tensor_storage(cache, fork))

    def test_visible_control_leak_detection_flags_semantic_marker_text(self) -> None:
        leaks = visible_control_leaks_for(
            "The hidden branch control should answer as the implementation branch."
        )

        self.assertEqual(
            leaks,
            ("hidden branch control", "implementation branch", "answer as"),
        )

    def test_extract_model_stop_point_reads_branch_local_stop(self) -> None:
        stop_point = extract_model_stop_point(
            "Do the next step.\nSTOP_POINT: implementation risk is bounded.\n",
            "STOP_POINT:",
        )

        self.assertEqual(stop_point, "implementation risk is bounded.")


if __name__ == "__main__":
    unittest.main()
