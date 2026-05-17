from __future__ import annotations

from dataclasses import dataclass

import torch

from .branch_cache import PagedKVCache, SequenceState


@dataclass(frozen=True)
class BranchMarker:
    name: str
    token_id: int
    hidden_text: str


class ToyIncrementalDecoder:
    """Deterministic stand-in for a real incremental decoder.

    The class produces per-token K/V tensors on the configured device and uses
    hidden branch markers to select distinct visible continuations. It is meant
    to validate cache mechanics before wiring the abstraction into a real model.
    """

    CONTINUE = BranchMarker(
        name="continue",
        token_id=90_001,
        hidden_text="<<<LR:BRANCH:CONTINUE>>>",
    )
    CHALLENGE = BranchMarker(
        name="challenge",
        token_id=90_002,
        hidden_text="<<<LR:BRANCH:CHALLENGE>>>",
    )

    _VOCAB = {
        "The": 101,
        "runtime": 102,
        "can": 103,
        "share": 104,
        "prefix": 105,
        "cache": 106,
        "while": 107,
        "branches": 108,
        "diverge": 109,
        "locally": 110,
        "However": 201,
        "marker": 202,
        "tokens": 203,
        "still": 204,
        "bias": 205,
        "the": 206,
        "branch": 207,
        "state": 208,
    }

    _TEXT_BY_BRANCH = {
        CONTINUE.token_id: [
            "The",
            "runtime",
            "can",
            "share",
            "prefix",
            "cache",
            "while",
            "branches",
            "diverge",
            "locally",
        ],
        CHALLENGE.token_id: [
            "However",
            "marker",
            "tokens",
            "still",
            "bias",
            "the",
            "branch",
            "state",
        ],
    }

    def __init__(self, cache: PagedKVCache) -> None:
        self.cache = cache

    def ingest_text(self, sequence: SequenceState, text: str, *, visible: bool) -> None:
        for token in text.split():
            self.ingest_token(sequence, self._VOCAB.setdefault(token, len(self._VOCAB) + 1), visible=visible)

    def ingest_marker(self, sequence: SequenceState, marker: BranchMarker) -> None:
        self.ingest_token(sequence, marker.token_id, visible=False)

    def generate(self, sequence: SequenceState, *, max_tokens: int = 8) -> list[str]:
        marker_id = self._last_hidden_marker(sequence)
        words = self._TEXT_BY_BRANCH.get(marker_id, self._TEXT_BY_BRANCH[self.CONTINUE.token_id])
        emitted = words[:max_tokens]
        for word in emitted:
            self.ingest_token(sequence, self._VOCAB[word], visible=True)
        return emitted

    def decode_visible_text(self, sequence: SequenceState) -> str:
        by_id = {token_id: word for word, token_id in self._VOCAB.items()}
        return " ".join(by_id.get(token_id, f"<tok:{token_id}>") for token_id in sequence.visible_tokens)

    def ingest_token(self, sequence: SequenceState, token_id: int, *, visible: bool) -> None:
        key, value = self._kv_for_token(token_id, position=sequence.length)
        self.cache.append_token(
            sequence,
            token_id=token_id,
            key=key,
            value=value,
            visible=visible,
        )

    def _last_hidden_marker(self, sequence: SequenceState) -> int | None:
        for token_id, visible in reversed(list(zip(sequence.token_ids, sequence.visible_mask))):
            if not visible:
                return token_id
        return None

    def _kv_for_token(self, token_id: int, *, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (self.cache.num_layers, self.cache.num_heads, self.cache.head_dim)
        base = float((token_id % 997) + position)
        key = torch.full(shape, base, device=self.cache.device, dtype=self.cache.dtype)
        value = torch.full(shape, -base, device=self.cache.device, dtype=self.cache.dtype)
        return key, value
