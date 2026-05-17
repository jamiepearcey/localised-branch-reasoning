# CUDA LoRA Training Process

## Purpose

The current demos use context instructions to make a model emit:

- a decision point
- bounded factors
- one continuation per factor
- a branch-local stop point
- evaluation and resolution decisions

The fine-tuning phase trains this behavior into the model so less of the
protocol has to be carried in the prompt.

## Recommended Technique

Use LoRA for larger CUDA budgets and QLoRA for constrained rented GPUs.

Start with QLoRA:

- base model: `Qwen/Qwen2.5-7B-Instruct` for the first training loop
- later base model: a 14B or 32B Qwen instruct/coder model if GPU memory allows
- adapter rank: `r=16`
- alpha: `32`
- dropout: `0.05`
- target modules: `all-linear`
- sequence length: `4096`
- bf16 when supported

QLoRA is a practical first pass because it keeps the frozen base model in 4-bit
NF4 and trains small LoRA adapters.

## Dataset

Build seed SFT data from completed planned-branch reports:

```bash
PYTHONPATH=src python3 scripts/build_branching_sft_dataset.py \
  --input reports/planned_branch_kv_demo_qwen_1_5b.txt \
  --output data/branching_sft_seed.jsonl
```

The generated format uses explicit training tags:

```text
<LR_TASK>...</LR_TASK>
<LR_DECISION_POINT>...</LR_DECISION_POINT>
<LR_FACTORS>build, review</LR_FACTORS>
<LR_BRANCH factor="build" id="build">
...
STOP_POINT: ...
</LR_BRANCH>
<LR_EVALUATE_BRANCHES/>
```

These tags are a training format, not a secrecy mechanism. They are used so the
model can learn the control structure before later experiments compress or
remove the visible protocol.

## Local Validation

Validate data on the Mac without CUDA:

```bash
python3 scripts/train_branching_lora.py \
  --dataset data/branching_sft_seed.jsonl \
  --dry-run
```

## CUDA Training

On a CUDA host:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U torch transformers datasets accelerate peft trl bitsandbytes

python3 scripts/train_branching_lora.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/branching_sft_seed.jsonl \
  --output-dir adapters/branching-lora \
  --qlora \
  --bf16 \
  --max-seq-length 4096 \
  --epochs 1
```

For a larger GPU, remove `--qlora` and train LoRA in bf16.

## Evaluation After Training

Run the same planned-branch demos with the adapter loaded. The first target
metric is not benchmark score; it is lower protocol overhead and better branch
discipline:

- planner emits parseable decision point and factors
- branches stay on their selected factor
- branches emit meaningful `STOP_POINT`
- evaluator chooses useful keep/revise/expand/discard decisions
- fewer tokens are spent on explicit prompt scaffolding

## Next Backend Work

This fine-tuning process is separate from the KV backend. The adapter can be
trained with ordinary Hugging Face SFT on CUDA. The runtime still needs a real
KV-fork backend, such as llama.cpp sequence memory on Metal or CUDA paged
attention later.
