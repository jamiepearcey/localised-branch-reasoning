# Localised Reasoning KV Fork Prototype

This repository starts with a small, runnable prototype of efficient KV-cache
forking.

The current implementation uses PyTorch tensors and selects `mps` when
available, which exercises Apple's Metal path on Apple Silicon. The code is
structured around cache-block ownership rather than PyTorch-specific APIs so
the same sequence-state model can later be mapped to CUDA paged attention.

## What This Tests

- Prefill a shared prefix into KV blocks.
- Fork sequence metadata without copying prefix KV tensors.
- Track `SequenceState.block_ids`, logical length, visible length, hidden
  marker length, and branch id.
- Track `KVBlock` storage, refcount, token capacity, used tokens, and explicit
  `shared` or `branch-local` ownership.
- Reference-count shared prefix blocks.
- Append different hidden continuation markers per branch.
- Allocate branch-local KV blocks only after divergence.
- Keep hidden marker tokens out of the visible transcript.
- Separately test generated visible output for semantic leakage of marker
  control phrases.

## Run

```bash
python3 scripts/kv_fork_demo.py
python3 -m unittest discover -s tests
```

Expected device on Apple Silicon:

```text
device=mps
```

## Real Model Demo

For a real `past_key_values` cache fork, use the Hugging Face demo:

```bash
PYTHONPATH=src python3 scripts/real_kv_fork_demo.py
PYTHONPATH=src python3 scripts/real_kv_fork_demo.py --output-file reports/real_kv_fork_demo.txt
```

This pre-fills the shared prefix once, creates two branch cache objects that
initially share the prefix KV tensor storage, ingests different hidden branch
markers, and verifies forked-cache logits against a full recompute of
`prefix + marker`.

Hidden marker tokens are excluded from the visible transcript and are not
replayed to the user as input text. However, because they are ordinary model
context tokens, the model may still refer to their content in generated output.
The marker protocol is therefore a behavioral steering mechanism, not a secrecy
or isolation mechanism.

The real KV demo also fails if visible output contains obvious control-language
phrases such as `hidden branch control`, `branch control marker`,
`implementation branch`, `runtime reviewer branch`, or `answer as`.

The important proof lines are:

- `prefix_forward_calls=1`
- `fork_shared_prefix_storage=True`
- `prefix_unchanged_after_branches=True`
- `fork_vs_full_recompute_logits_max_abs_diff` near zero

The default model is `Qwen/Qwen2.5-0.5B-Instruct` with eager attention, which
works on MPS in this prototype. For a fully offline smoke test using the cached
GPT-2 model:

```bash
PYTHONPATH=src python3 scripts/real_kv_fork_demo.py --model gpt2 --local-files-only
```

For the first real local LLM run, use Ollama with a small Qwen3 GGUF model:

```bash
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
PYTHONPATH=src python3 scripts/real_llm_branch_demo.py
```

This validates the branch-marker protocol against an actual model on Apple
Silicon. Ollama does not expose KV cache block tables, so this is a behavioral
test, not the efficient KV-fork implementation.

To compare marker behavior across multiple prompts:

```bash
PYTHONPATH=src python3 scripts/evaluate_branch_markers.py
PYTHONPATH=src python3 scripts/evaluate_branch_markers.py --seeds 11,12,13 --strict
PYTHONPATH=src python3 scripts/evaluate_branch_markers.py --strict --json-output reports/branch_marker_eval.json
```

Both real-model scripts print a lightweight assessment for each continuation.
The checks catch marker leakage, invalid fork mechanics, and branch-role collapse.

The current branch-control invariants are documented in
[`docs/protocol.md`](docs/protocol.md).

## Planner-Decided Branches

The next demo lets the model choose the decision point and factors from a
bounded catalog, then rewrites the stacked factor row into one real KV branch
per factor:

```bash
PYTHONPATH=src python3 scripts/planned_branch_kv_demo.py \
  --output-file reports/planned_branch_kv_demo.txt
```

For the cached 1.5B Qwen run used during development:

```bash
PYTHONPATH=src python3 scripts/planned_branch_kv_demo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local-files-only \
  --continuation-max-new-tokens 28 \
  --output-file reports/planned_branch_kv_demo_qwen_1_5b.txt
```

The report shows:

- the planner prompt with the bounded factor catalog
- the raw model-emitted `DECISION_POINT` and `FACTORS`
- the stacked marker before rewrite
- the rewritten shared prefix with the stacked factor row removed
- one rewritten hidden marker per selected factor
- the KV fork proof lines, visible continuation, and branch-local
  `STOP_POINT` for each factor branch

Each branch decides its own stop point. The runtime streams tokens until that
branch emits `STOP_POINT:` or reaches the safety token cap, then records
`model_stop_detected` and `model_stop_point` in the report.

## Multi-Branch Evaluation

The next process is documented in
[`docs/multi_branch_evaluation.md`](docs/multi_branch_evaluation.md). It treats
completed branches as immutable artifacts, evaluates each branch for relevance,
novelty, risk, and actionability, then emits a resolution:

```text
merge | continue_branch | fork_branch | replan | stop
```

The evaluation schema lives in
[`src/localised_reasoning/branch_evaluation.py`](src/localised_reasoning/branch_evaluation.py).

## CUDA Fine-Tuning

The CUDA adapter-training plan is documented in
[`docs/cuda_lora_training.md`](docs/cuda_lora_training.md). The initial path is
QLoRA/LoRA SFT so the model learns the branching protocol instead of relying
entirely on context scaffolding.

Build seed SFT data from completed branch reports:

```bash
PYTHONPATH=src python3 scripts/build_branching_sft_dataset.py
```

Validate locally without CUDA:

```bash
python3 scripts/train_branching_lora.py --dry-run
```

Run on CUDA:

```bash
pip install -e ".[cuda-train]"
python3 scripts/train_branching_lora.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/branching_sft_seed.jsonl \
  --output-dir adapters/branching-lora \
  --qlora \
  --bf16
```

## 30B-Class Metal Path

For the Mac target, the practical local route is llama.cpp with GGUF
quantization on Metal. This repository now includes a C++ llama.cpp sequence
fork demo:

```bash
brew install llama.cpp
scripts/build_llama_seq_fork_demo.sh
scripts/build_llama_branch_worker.sh
```

Recommended first 30B-class GGUF targets on a 36 GB unified-memory M4 Max are:

- [`Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF)
  for the first planned-branch run because it is a single 30B-total MoE GGUF
  with lower active parameters.
- [`Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF)
  for coding-oriented continuation quality.
- [`bartowski/Qwen2.5-32B-Instruct-GGUF:Q4_K_M`](https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF)
  for a general instruct 32B option.

Use `Q4_K_M` first. It leaves room for model overhead, Metal buffers, the KV
cache, and multiple branch sequences. `Q5_K_M` may fit at short context lengths
but gives less headroom for branch work.

Run a normal llama.cpp smoke test first:

```bash
llama-cli -hf Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M \
  -ngl 999 -c 4096 \
  -p "Give one concrete next step for a KV-cache branching runtime."
```

Download the first GGUF file to an explicit local path for the branch demo:

```bash
python3 -m pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF \
  Qwen3-30B-A3B-Q4_K_M.gguf \
  --local-dir models/qwen3-30b-a3b-q4_k_m
```

Then run the planned branch demo against the downloaded `.gguf` file:

```bash
build/llama_seq_fork_demo \
  --model models/qwen3-30b-a3b-q4_k_m/Qwen3-30B-A3B-Q4_K_M.gguf \
  --planned \
  --ctx-size 4096 \
  --gpu-layers 999 \
  --planner-max-new-tokens 128 \
  --max-new-tokens 96 \
  --stop-extra-tokens 48 \
  --output-file reports/llama_seq_fork_demo.txt
```

The important proof lines are:

- `prefix_forward_passes=1`
- `sequence_copy_api=llama_memory_seq_cp`
- `planner_selected_factors=[...]`
- each `seq_after_copy_pos_max` equals the prefix end
- `seq0_final_pos_max` remains the prefix end after branch generation
- each branch has `model_stop_detected=true`
- each branch has `visible_control_leaks=[]`

This is the right next local proof for quantized 30B-class models on Metal. It
uses llama.cpp's real sequence-level KV memory APIs, so the model is not
recomputing the shared prefix per branch. It is still not the final
device-independent paged COW block table. The remaining engineering question is
how much memory `llama_memory_seq_cp` duplicates internally for this model and
cache layout; that has to be measured before claiming COW-equivalent prefix
sharing.

To run the black-box marker-bias probe:

```bash
PYTHONPATH=src python3 scripts/probe_marker_bias.py --strict
PYTHONPATH=src python3 scripts/probe_marker_bias.py --seeds 21,22 --strict
PYTHONPATH=src python3 scripts/probe_marker_bias.py --strict --json-output reports/marker_bias_probe.json
```

To run the current regression gate:

```bash
PYTHONPATH=src python3 scripts/run_all_checks.py
```

This runs unit tests, compiles the llama.cpp sequence-fork demo and live branch
worker, runs the real Hugging Face KV-fork check, runs the two-seed
branch-marker evaluation, and runs the two-seed marker-bias probe. It writes
JSON summaries under `reports/`, which is ignored by git because these are run
artifacts. Use
`--skip-llama-cpp-build` only on machines without the Homebrew llama.cpp
headers.

## Python Q&A Scenario Pipeline

The Q&A scenario workflow is intentionally Python-owned. C++ remains the
low-level llama.cpp KV-fork primitive, while Python owns question sets, scenario
definitions, DataFrame assembly, adjudication results, and spreadsheet export.

For a cheap smoke test without model inference:

```bash
PYTHONPATH=src python3 scripts/run_qa_scenarios.py --limit 3 --no-xlsx
```

This writes `reports/qa_scenario_branches.csv` with one question per row and one
answer column per scenario branch. Excel export is available through:

```bash
PYTHONPATH=src python3 scripts/run_qa_scenarios.py --limit 5
```

Install the `qa-report` optional dependency first if the local environment does
not already have an Excel writer:

```bash
python3 -m pip install '.[qa-report]'
```

## Comparative Evaluation

The blind comparison workflow scores reasoned branch answers selected by a
small non-solving selector against a single-pass reasoning baseline with the
same total output-token budget. The answer key is used only by the scorer, not
by the generation interface.

For a cheap proxy run that exercises the full workbook process without model
inference:

```bash
PYTHONPATH=src python3 scripts/run_comparative_eval.py
```

This writes `reports/comparative_eval_proxy.xlsx` with:

- `Summary`: branch accuracy, reasoning accuracy, branch-only wins,
  reasoning-only wins, and branch-hurt counts
- `Per Question`: scored paired comparison for each question
- `Branch Raw`: every scenario branch answer
- `Reasoning Raw`: the non-branching baseline answer under the same token budget

The default reasoning budget is:

```text
5 reasoned scenario continuations * 160 tokens + 48 selector tokens = 848 tokens
```

The proxy engine is deliberately not a quality claim. It creates branch wins,
reasoning wins, and branch-hurt cases so the scoring process can be inspected
before spending on a real model run.

Scoring uses deterministic answer patterns for proxy runs. For real worker runs,
`--scorer auto` uses the resident model as an answer-equivalence scorer after
generation: the scorer receives the question, the actual answer, and the given
answer, then emits `CORRECT`, `CONFIDENCE`, and `RATIONALE`. Override with
`--scorer regex` for the old deterministic scorer or `--scorer worker` to force
LLM scoring.

For a larger public benchmark, use MMLU-Pro. It provides a 12k-row test split
with 10-option multiple-choice questions, categories, and deterministic answer
labels. The loader uses the Hugging Face datasets-server API and normalizes
rows into this repo's existing `EvalQuestion` schema.

Prepare an inspectable CSV slice:

```bash
PYTHONPATH=src python3 scripts/prepare_benchmark_dataset.py \
  --source mmlu-pro \
  --split test \
  --limit 500 \
  --shuffle-seed 123 \
  --output-csv data/benchmarks/mmlu_pro_500.csv
```

Run the comparative process against an MMLU-Pro slice:

```bash
PYTHONPATH=src python3 scripts/run_comparative_eval.py \
  --engine worker \
  --question-source mmlu-pro \
  --benchmark-split test \
  --benchmark-shuffle-seed 123 \
  --limit 100 \
  --model-preset qwen3-30b-a3b \
  --output-xlsx reports/comparative_eval_mmlu_pro_100.xlsx
```

Available local GGUF presets are:

- `qwen3-30b-a3b`: current default, highest-quality local preset.
- `qwen3-14b`: middle option for faster benchmark iteration.
- `qwen3-4b`: small fast option for testing the same branch/gate pipeline.

Download the smaller presets with:

```bash
huggingface-cli download unsloth/Qwen3-14B-GGUF \
  Qwen3-14B-Q4_K_M.gguf \
  --local-dir models/qwen3-14b-q4_k_m

huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF \
  Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --local-dir models/qwen3-4b-instruct-2507-q4_k_m
```

Use `--model-path` when you want to override the preset with another GGUF file.

For MMLU-Pro, the runner uses benchmark-specific branch roles rather than the
generic trap-question roles. The roles are deliberately operational rather than
stylistic: direct derivation, source-of-truth recall, option backsolving,
counterexample testing, formula/unit checks, and distractor elimination. The
report also includes deterministic branch diagnostics: unique answer count,
thin evidence count, strong evidence count, arithmetic checks, contradiction
signals, answer/evidence mismatches, and whether the branch set collapsed to
one weakly supported answer. The baseline-aware meta-gate uses those diagnostics
and chooses between the branch selector candidate and the monolithic reasoning
candidate. If all branches collapse to one thin answer, the explicit diversity
gate prefers the baseline unless the branch and baseline already agree.

The benchmark runner can also use a controlled branch taxonomy and let the
resident model choose the branch roles per question. This is bounded: the model
chooses from named operations such as `question_target_filter`,
`formula_mapper`, `unit_conversion_checker`, `option_backsolver`,
`magnitude_estimator`, `source_of_truth_recall`, `rule_elements`,
`exception_checker`, `distractor_eliminator`, and
`evidence_answer_auditor`; it does not invent free-form roles.

```bash
PYTHONPATH=src python3 scripts/run_comparative_eval.py \
  --engine worker \
  --question-source mmlu-pro \
  --limit 25 \
  --model-preset qwen3-30b-a3b \
  --scenario-mode taxonomy-model \
  --branch-layout late-question-fork \
  --output-xlsx reports/comparative_eval_taxonomy_model.xlsx
```

`--scenario-mode fixed` keeps the older fixed five-role benchmark.
`--scenario-mode taxonomy-category` uses a deterministic category-to-role
mapping without paying for model role selection. `--scenario-mode
taxonomy-model` asks the loaded model to choose 3-5 roles from the taxonomy for
each question. The selected roles are written to `selected_branch_roles` and
`selected_branch_count` in the per-question output.

`--branch-layout late-question-fork` is the original efficient path:

```text
static prefix -> question -> KV fork -> branch operation
```

`--branch-layout role-before-question` keeps static-prefix KV reuse but forks
before the branch operation and question:

```text
static prefix -> KV fork -> branch operation + question
```

The second layout costs more question-token compute and is not the primary
branch-continuation architecture. Use it only as a diagnostic accuracy probe to
test whether earlier role conditioning escapes a bad shared question
interpretation.

You can restrict by broad MMLU-Pro categories:

```bash
PYTHONPATH=src python3 scripts/run_comparative_eval.py \
  --engine worker \
  --question-source mmlu-pro \
  --benchmark-categories "computer science,math,physics" \
  --limit 100 \
  --model-preset qwen3-4b
```

To compare the same benchmark rows across local model presets:

```bash
PYTHONPATH=src python3 scripts/run_model_preset_matrix.py \
  --benchmark-offset 0 \
  --limit 50 \
  --presets qwen3-4b,qwen3-14b,qwen3-30b-a3b \
  --output-xlsx reports/model_preset_matrix_000_050.xlsx
```

To run the same comparative process against the local llama.cpp Metal model,
opt in explicitly:

```bash
PYTHONPATH=src python3 scripts/run_comparative_eval.py \
  --engine worker \
  --model-preset qwen3-30b-a3b \
  --limit 3 \
  --output-xlsx reports/comparative_eval_worker.xlsx
```

In this mode, Python owns orchestration and scoring. Branch answers use the
resident `build/llama_branch_worker` process instead of reloading the model per
question. The worker loads the GGUF once, accepts JSONL requests over
stdin/stdout, prefills the shared prefix once per branch request, forks it with
`llama_memory_seq_cp`, appends one hidden marker per branch, and advances active
branches in a batched lockstep decode loop. The final selector uses the same
resident worker in non-branching `generate` mode but is instructed to choose
only from supplied branch outputs, not to solve from scratch. The reasoning
baseline also uses the resident worker and receives the full branching compute
budget. The meta-gate is another non-branching `generate` request with a fresh
KV state inside the already-loaded worker process; it sees the visible branch
outputs, the branch selector candidate, and the baseline reasoning candidate,
then chooses between those candidates. It does not reuse branch KV state and it
is not a deterministic verifier. The answer key remains hidden until the
scoring step.

For repeated benchmark runs, the live worker also supports resident static
prefix caches. Python registers reusable branch-instruction, selector, and
meta-gate prefixes once with `cache_prefix`, then sends only the per-question
suffix through `cached_branch` or `cached_generate`. This avoids recomputing
stable instruction tokens while still giving each question and branch fresh
working sequence state.

## Scope

The Hugging Face path is a real model KV-cache fork using `past_key_values`.
Because `DynamicCache` appends with tensor concatenation, each branch is
materialized after marker ingestion. That is fast enough to avoid repeated
prefix prefill and prove correctness, but it is not yet a paged-attention
copy-on-write runtime.

The llama.cpp path is a real quantized-GGUF Metal path using
`llama_memory_seq_cp` to copy one sequence's KV memory into branch sequence IDs.
This requires llama.cpp's unified KV buffer mode in the current Homebrew build;
per-sequence KV layout aborts on `llama_memory_seq_cp`. It is intended for local
30B-class testing on Apple Silicon before a CUDA paged-attention backend is
added. The single-shot `llama_seq_fork_demo` is the proof report generator; the
persistent `llama_branch_worker` is the Python-facing live API for comparative
experiments.

## Long Decision Localisation

The short-answer benchmark is not the strongest test of localized reasoning.
For the architecture's intended use case, use the synthetic long-decision
workflow:

```bash
PYTHONPATH=src python3 scripts/run_long_decision_eval.py \
  --engine worker \
  --model-preset qwen3-30b-a3b \
  --ctx-size 4096 \
  --limit 3 \
  --output-xlsx reports/long_decision_30b_aggregate_000_003.xlsx
```

Each case contains a long decision context and a bounded factor catalog. The
resident model first decides whether to branch and which factors to evaluate.
The runtime then uses the efficient continuation architecture:

```text
static prefix + case context + planner decision -> KV fork -> one factor marker per branch
completed factor artifacts -> final aggregation pass -> one collapsed decision
```

The comparison baselines are:

- `forked_branch`: factor continuations from a shared case KV state
- `forked_aggregate`: final decision collapsed from completed independent
  factor artifacts
- `sequential`: a true accumulating context where each factor sees previous
  factor outputs
- `monolithic`: one integrated decision pass

The output workbook contains planner choices, raw forked branch outputs,
the forked and sequential aggregate decisions, monolithic answers, deterministic
focus/contamination metrics, and an LLM judge sheet. The judge scores final
decisions on groundedness, factor coverage, synthesis quality, risk handling,
actionability, and overall quality. The contamination scorer ignores explicit
`TRADEOFF_BOUNDARY` text so branches are not penalized for naming what they
intentionally did not decide.

## Coding Checkpoint Branching

For coding tasks, use the multi-checkpoint branch/collapse workflow:

```bash
PYTHONPATH=src python3 scripts/run_coding_branch_eval.py \
  --engine worker \
  --model-preset qwen3-30b-a3b \
  --ctx-size 4096 \
  --limit 2 \
  --consideration-limit 4 \
  --output-xlsx reports/coding_branch_30b_000_002.xlsx
```

This workflow asks the model to choose high-risk method checkpoints, with the
planner prompt explicitly preferring method start, before mutation, external
side-effect boundaries, loops/batches, and method end. At each selected
checkpoint, the runtime forks localized branches for coding considerations such
as contract, edge cases, state consistency, security, performance, and tests.
Those branches are collapsed into a checkpoint artifact. The final pass then
collapses all checkpoint artifacts into one patch/test plan.

The efficient shape is:

```text
code context + active checkpoint -> KV fork -> consideration branches
consideration artifacts -> checkpoint collapse
checkpoint collapses -> final patch plan
```

The report includes selected checkpoints, branch outputs, checkpoint collapses,
the final branch-derived patch plan, a monolithic comparator, and simple
checkpoint/locality metrics.

## SWE-bench Lite / Verified

The SWE-bench runner applies the same checkpoint branch/collapse architecture
to real SWE-bench Lite or Verified instances:

```bash
PYTHONPATH=src python3 scripts/run_swebench_branch_eval.py \
  --engine worker \
  --dataset lite \
  --model-preset qwen3-30b-a3b \
  --ctx-size 8192 \
  --limit 1 \
  --max-files 2 \
  --output-dir reports/swebench_lite_branch_000_001 \
  --output-xlsx reports/swebench_lite_branch_000_001.xlsx
```

The runner fetches the public SWE-bench row from Hugging Face, uses the gold
patch only to identify file paths for source context, then fetches the base
commit source from GitHub raw URLs. The gold patch and test patch are not
included in the model prompt. This is an oracle-file-localization setup, not a
fully autonomous repository agent.

The output directory contains two harness-compatible prediction files:

```text
swebench_branch_predictions.jsonl
swebench_monolithic_predictions.jsonl
```

Each row uses the official SWE-bench prediction shape:

```json
{"instance_id": "...", "model_name_or_path": "...", "model_patch": "diff --git ..."}
```

To score a generated file with the official Docker harness:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path reports/swebench_lite_branch_000_001/swebench_branch_predictions.jsonl \
  --instance_ids <instance_id> \
  --max_workers 1 \
  --run_id localised-branch-lite-000-001
```

On Apple Silicon, SWE-bench documents ARM evaluation as experimental and notes
that `--namespace ''` forces local image builds instead of pulling Linux images.
For serious pass/fail runs, use a Linux Docker host or a cloud runner.
