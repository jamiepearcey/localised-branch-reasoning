#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localised_reasoning import PagedKVCache, ToyIncrementalDecoder  # noqa: E402


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = choose_device()
    cache = PagedKVCache(
        num_layers=2,
        num_heads=4,
        head_dim=16,
        block_size=4,
        device=device,
    )
    decoder = ToyIncrementalDecoder(cache)

    prefix = cache.new_sequence("prefix")
    decoder.ingest_text(prefix, "The runtime can share prefix cache", visible=True)

    branch_continue = cache.fork(prefix, name="continue")
    branch_challenge = cache.fork(prefix, name="challenge")

    decoder.ingest_marker(branch_continue, ToyIncrementalDecoder.CONTINUE)
    decoder.ingest_marker(branch_challenge, ToyIncrementalDecoder.CHALLENGE)

    continue_words = decoder.generate(branch_continue, max_tokens=8)
    challenge_words = decoder.generate(branch_challenge, max_tokens=8)

    shared_prefix_ids = [
        block.block_id
        for block in branch_continue.block_table
        if block in branch_challenge.block_table
    ]
    continue_ids = [block.block_id for block in branch_continue.block_table]
    challenge_ids = [block.block_id for block in branch_challenge.block_table]

    print(f"device={device}")
    print(f"prefix_length={prefix.length}")
    print(f"continue_length={branch_continue.length}")
    print(f"challenge_length={branch_challenge.length}")
    print(f"shared_prefix_blocks={shared_prefix_ids}")
    print(f"continue_blocks={continue_ids}")
    print(f"challenge_blocks={challenge_ids}")
    print(f"allocated_token_slots={cache.allocated_token_slots()}")
    print(f"continue_hidden_tokens={branch_continue.hidden_tokens}")
    print(f"challenge_hidden_tokens={branch_challenge.hidden_tokens}")
    print(f"continue_generated={' '.join(continue_words)}")
    print(f"challenge_generated={' '.join(challenge_words)}")
    print(f"continue_visible={decoder.decode_visible_text(branch_continue)}")
    print(f"challenge_visible={decoder.decode_visible_text(branch_challenge)}")


if __name__ == "__main__":
    main()
