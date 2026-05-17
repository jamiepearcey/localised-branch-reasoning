from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import torch


@dataclass
class KVBlock:
    """A fixed-size block of cached keys and values."""

    block_id: int
    keys: torch.Tensor
    values: torch.Tensor
    used: int = 0
    refcount: int = 1
    owner: Literal["shared", "branch-local"] = "branch-local"
    owner_branch: str | None = None

    @property
    def storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[2])

    @property
    def capacity_tokens(self) -> int:
        return self.capacity

    @property
    def used_tokens(self) -> int:
        return self.used

    @property
    def is_full(self) -> bool:
        return self.used >= self.capacity


@dataclass
class SequenceState:
    """Logical sequence state with visible and hidden token accounting."""

    name: str
    branch_id: str
    token_ids: list[int] = field(default_factory=list)
    visible_mask: list[bool] = field(default_factory=list)
    block_table: list[KVBlock] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def logical_length(self) -> int:
        return self.length

    @property
    def visible_length(self) -> int:
        return sum(1 for visible in self.visible_mask if visible)

    @property
    def hidden_prefix_length(self) -> int:
        return self.logical_length - self.visible_length

    @property
    def block_ids(self) -> list[int]:
        return [block.block_id for block in self.block_table]

    @property
    def visible_tokens(self) -> list[int]:
        return [
            token_id
            for token_id, visible in zip(self.token_ids, self.visible_mask)
            if visible
        ]

    @property
    def hidden_tokens(self) -> list[int]:
        return [
            token_id
            for token_id, visible in zip(self.token_ids, self.visible_mask)
            if not visible
        ]


class PagedKVCache:
    """Reference-counted block KV cache with copy-on-append branching.

    Forking copies only sequence metadata and block references. Once a sequence
    diverges, appending to a shared block allocates a new branch-local block
    rather than mutating the shared prefix block.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self._next_block_id = 0

    def new_sequence(self, name: str) -> SequenceState:
        return SequenceState(name=name, branch_id=name)

    def fork(self, parent: SequenceState, *, name: str) -> SequenceState:
        child = SequenceState(
            name=name,
            branch_id=name,
            token_ids=parent.token_ids.copy(),
            visible_mask=parent.visible_mask.copy(),
            block_table=parent.block_table.copy(),
        )
        for block in child.block_table:
            block.refcount += 1
            block.owner = "shared"
            block.owner_branch = None
        return child

    def release(self, sequence: SequenceState) -> None:
        for block in sequence.block_table:
            block.refcount -= 1
            if block.refcount < 0:
                raise RuntimeError(f"KV block {block.block_id} refcount underflow")
        sequence.block_table.clear()
        sequence.token_ids.clear()
        sequence.visible_mask.clear()

    def append_token(
        self,
        sequence: SequenceState,
        *,
        token_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
        visible: bool,
    ) -> None:
        key = self._normalize_kv(key, name="key")
        value = self._normalize_kv(value, name="value")

        block = self._appendable_block(sequence)
        offset = block.used
        block.keys[:, :, offset, :] = key
        block.values[:, :, offset, :] = value
        block.used += 1

        sequence.token_ids.append(int(token_id))
        sequence.visible_mask.append(bool(visible))

    def append_tokens(
        self,
        sequence: SequenceState,
        tokens: Iterable[tuple[int, torch.Tensor, torch.Tensor, bool]],
    ) -> None:
        for token_id, key, value, visible in tokens:
            self.append_token(
                sequence,
                token_id=token_id,
                key=key,
                value=value,
                visible=visible,
            )

    def allocated_token_slots(self) -> int:
        return self._next_block_id * self.block_size

    def _appendable_block(self, sequence: SequenceState) -> KVBlock:
        if (
            not sequence.block_table
            or sequence.block_table[-1].is_full
            or sequence.block_table[-1].refcount > 1
            or sequence.block_table[-1].owner == "shared"
        ):
            block = self._allocate_block(owner_branch=sequence.branch_id)
            sequence.block_table.append(block)
            return block
        if sequence.block_table[-1].owner == "shared":
            raise RuntimeError(
                f"attempted to append to shared KV block {sequence.block_table[-1].block_id}"
            )
        return sequence.block_table[-1]

    def _allocate_block(self, *, owner_branch: str) -> KVBlock:
        shape = (self.num_layers, self.num_heads, self.block_size, self.head_dim)
        block = KVBlock(
            block_id=self._next_block_id,
            keys=torch.empty(shape, device=self.device, dtype=self.dtype),
            values=torch.empty(shape, device=self.device, dtype=self.dtype),
            owner="branch-local",
            owner_branch=owner_branch,
        )
        self._next_block_id += 1
        return block

    def _normalize_kv(self, tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        expected = (self.num_layers, self.num_heads, self.head_dim)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        if tensor.dtype != self.dtype:
            tensor = tensor.to(self.dtype)
        return tensor
