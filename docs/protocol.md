# Branch-Control Protocol Notes

The current prototype treats branch controls as ordinary text markers that are
inserted into model input by the runtime. The UI transcript must not show those
markers, and the shared prefix must not contain them.

Hidden marker tokens are excluded from the visible transcript and are not
replayed to the user as input text. However, because they are ordinary model
context tokens, the model may still refer to their content in generated output.
The marker protocol is therefore a behavioral steering mechanism, not a secrecy
or isolation mechanism.

## Runtime Invariants

The target device-independent block-table abstraction is:

```text
SequenceState:
  block_ids: list[int]
  logical_length: int
  visible_length: int
  hidden_prefix_length: int
  branch_id: str

KVBlock:
  storage: tensor handles
  refcount: int
  capacity_tokens: int
  used_tokens: int
  owner: shared | branch-local
```

- Forks happen only at completed token boundaries.
- Sequence metadata is shallow-copied at the fork point.
- Prefix KV blocks are shared by reference and treated as immutable.
- `fork(prefix).block_ids == prefix.block_ids`, and prefix block refcounts
  increment on fork.
- The hidden marker is the first branch-local model input after the fork.
- Marker tokens are processed by the model and allocate branch-local suffix KV
  blocks.
- Appending to one branch must not change another branch's `block_ids`.
- Appending to one branch allocates only new tail blocks.
- Writes to shared blocks are disallowed or must trigger copy-on-write.
- Visible continuation tokens are generated after marker ingestion.
- Visible continuation tokens are conditioned by shared prefix plus
  branch-local suffix state.
- Visible continuation tokens may attend to the shared prefix.
- Visible continuation tokens must append only to branch-local suffix KV blocks.
- Generated tokens are values, and KV blocks are storage; runtime cache
  operations are the actors that append or write KV entries.
- Cache metadata, refcounts, ownership flags, and write counters are the source
  of truth for memory isolation.
- Mechanical invisibility and behavioral non-leakage are separate checks:
  hidden marker tokens must be excluded from the visible transcript, while
  generated visible output must be tested separately for semantic leakage of
  marker content.

## Prompt Invariants

- The model must not print marker text, examples, placeholder strings, or
  control block names.
- Prompt wording should avoid semantically rich marker text when the goal is to
  reduce visible references to the control mechanism.
- Regression checks should flag visible output that mentions obvious
  control-language phrases, including `hidden branch control`, `branch control
  marker`, `implementation branch`, `runtime reviewer branch`, and `answer as`.
- Branch outputs should be one inline paragraph with no line breaks, bullets,
  lists, or Markdown formatting.
- The builder branch should produce an implementation-oriented next step.
- The reviewer branch should produce a concrete failure mode and exposing test.
- Reviewer exposing tests should name cache metadata, block ownership, refcounts,
  or write counters as the evidence source.
- No-marker and inert-user-text conditions should remain semantically
  equivalent.
- Trusted-runtime-control output should follow the active branch instruction,
  so it is not expected to match no-marker output.

## Current Test Boundary

The Hugging Face `real_kv_fork_demo.py` path verifies a real model
`past_key_values` fork: one prefix prefill, shared prefix tensor storage before
branch append, marker ingestion per branch, and forked-cache logits checked
against full recompute of `prefix + marker`. The toy decoder verifies the local
block-table and copy-on-append mechanics on `mps`, `cuda`, or `cpu`. The Ollama
checks are behavioral tests only; Ollama does not expose KV block ownership, so
those checks cannot prove efficient cache forking.

The llama.cpp `llama_seq_fork_demo` path targets quantized 30B-class GGUF
models on Metal. It pre-fills sequence `0`, calls `llama_memory_seq_cp` to copy
the prefix KV memory into branch sequence IDs, appends different hidden marker
tokens per branch, and generates visible continuations from those branch-local
sequence states. This proves prefix reuse through llama.cpp's public sequence
memory API, but it does not by itself prove refcounted copy-on-write prefix
sharing. Memory growth and copy cost must be measured separately before treating
it as equivalent to the target block-table runtime.
