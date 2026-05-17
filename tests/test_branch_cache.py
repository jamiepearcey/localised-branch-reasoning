from __future__ import annotations

import unittest

import torch

from localised_reasoning import PagedKVCache, ToyIncrementalDecoder


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BranchCacheTests(unittest.TestCase):
    def test_fork_shares_prefix_blocks_and_allocates_divergent_suffix(self) -> None:
        cache = PagedKVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            block_size=4,
            device=device(),
        )
        decoder = ToyIncrementalDecoder(cache)

        prefix = cache.new_sequence("prefix")
        decoder.ingest_text(prefix, "The runtime can share prefix cache", visible=True)

        left = cache.fork(prefix, name="left")
        right = cache.fork(prefix, name="right")

        self.assertEqual(left.block_ids, prefix.block_ids)
        self.assertEqual(right.block_ids, prefix.block_ids)
        self.assertEqual(left.branch_id, "left")
        self.assertEqual(right.branch_id, "right")
        self.assertIs(left.block_table[0], right.block_table[0])
        self.assertGreater(left.block_table[0].refcount, 1)
        self.assertEqual(left.block_table[0].owner, "shared")

        right_block_ids_before = right.block_ids.copy()
        decoder.ingest_marker(left, ToyIncrementalDecoder.CONTINUE)
        decoder.ingest_marker(right, ToyIncrementalDecoder.CHALLENGE)

        self.assertIs(left.block_table[0], right.block_table[0])
        self.assertIsNot(left.block_table[-1], right.block_table[-1])
        self.assertEqual(right.block_ids[: len(right_block_ids_before)], right_block_ids_before)
        self.assertEqual(left.block_table[-1].owner, "branch-local")
        self.assertEqual(left.block_table[-1].owner_branch, "left")
        self.assertEqual(right.block_table[-1].owner, "branch-local")
        self.assertEqual(right.block_table[-1].owner_branch, "right")
        self.assertEqual(left.hidden_tokens, [ToyIncrementalDecoder.CONTINUE.token_id])
        self.assertEqual(right.hidden_tokens, [ToyIncrementalDecoder.CHALLENGE.token_id])
        self.assertEqual(left.logical_length, prefix.logical_length + 1)
        self.assertEqual(left.hidden_prefix_length, 1)

    def test_hidden_markers_do_not_appear_in_visible_transcript(self) -> None:
        cache = PagedKVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            block_size=4,
            device=device(),
        )
        decoder = ToyIncrementalDecoder(cache)
        sequence = cache.new_sequence("branch")

        decoder.ingest_text(sequence, "The runtime", visible=True)
        decoder.ingest_marker(sequence, ToyIncrementalDecoder.CHALLENGE)
        decoder.generate(sequence, max_tokens=4)

        visible = decoder.decode_visible_text(sequence)
        self.assertIn("The runtime", visible)
        self.assertIn("However marker tokens still", visible)
        self.assertNotIn("LR:BRANCH", visible)
        self.assertNotIn(str(ToyIncrementalDecoder.CHALLENGE.token_id), visible)

    def test_append_to_branch_does_not_mutate_other_branch_block_table(self) -> None:
        cache = PagedKVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            block_size=8,
            device=device(),
        )
        decoder = ToyIncrementalDecoder(cache)
        prefix = cache.new_sequence("prefix")
        decoder.ingest_text(prefix, "The runtime can", visible=True)

        branch_a = cache.fork(prefix, name="branch_a")
        branch_b = cache.fork(prefix, name="branch_b")
        shared_block = branch_a.block_table[-1]
        shared_used_before = shared_block.used_tokens
        branch_b_block_ids_before = branch_b.block_ids.copy()

        decoder.ingest_marker(branch_a, ToyIncrementalDecoder.CONTINUE)

        self.assertEqual(branch_b.block_ids, branch_b_block_ids_before)
        self.assertEqual(shared_block.used_tokens, shared_used_before)
        self.assertEqual(branch_a.block_ids[:-1], prefix.block_ids)
        self.assertNotIn(branch_a.block_ids[-1], prefix.block_ids)
        self.assertEqual(branch_a.block_table[-1].used_tokens, 1)

    def test_release_decrements_refcounts(self) -> None:
        cache = PagedKVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            block_size=4,
            device=device(),
        )
        decoder = ToyIncrementalDecoder(cache)
        prefix = cache.new_sequence("prefix")
        decoder.ingest_text(prefix, "The runtime", visible=True)

        child = cache.fork(prefix, name="child")
        shared = prefix.block_table[0]
        self.assertEqual(shared.refcount, 2)

        cache.release(child)
        self.assertEqual(shared.refcount, 1)


if __name__ == "__main__":
    unittest.main()
