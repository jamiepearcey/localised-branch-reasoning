# Multi-Branch Evaluation Process

## Goal

After the model chooses a decision point, factors, branch continuations, and
branch-local stop points, a separate evaluator process compares completed
branches without mutating their KV states.

## Inputs

```text
task
decision_point
selected_factors
branches:
  branch_id
  factor
  visible_output
  stop_point
  cache_metadata_summary
```

The evaluator consumes text artifacts and metadata. It must not append to or
overwrite any completed branch KV cache.

## Per-Branch Evaluation

Each branch receives:

```text
relevance: 1-5
novelty: 1-5
correctness_risk: 1-5
actionability: 1-5
decision: keep | revise | expand | discard
rationale: short text
```

The evaluator should mark a branch `revise` if the factor was useful but the
continuation leaked control syntax, repeated the prompt, stopped too early, or
missed the selected factor. It should mark a branch `expand` if the branch is
promising but needs a second fork from its stop point.

## Resolution

The evaluator then emits:

```text
decision: merge | continue_branch | fork_branch | replan | stop
selected_branch_ids: list[str]
merged_summary: str
next_prompt: str
```

The runtime acts on the resolution:

- `merge`: combine selected branch outputs into the final answer or next shared
  prefix.
- `continue_branch`: resume one stored branch KV state from its stop point.
- `fork_branch`: fork one stored branch KV state into a new factor set.
- `replan`: ask the planner for a fresh decision point and factor set.
- `stop`: finish with the merged summary.

## Training Signal

The best training examples contain both successful and failed branches. Useful
labels include:

- planner chose parseable factors without fallback
- each branch emitted a useful stop point
- each branch stayed on its factor
- evaluator kept, revised, expanded, or discarded the branch
- resolution chose the next runtime action

This lets a later LoRA learn not only the syntax, but also when branching was
worthwhile and when a branch should terminate.
