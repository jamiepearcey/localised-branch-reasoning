# Branching Training Data

Generated JSONL files in this directory are ignored by git.

The expected supervised fine-tuning format is one JSON object per line:

```json
{"text": "<LR_TASK>...</LR_TASK>\n<LR_DECISION_POINT>...</LR_DECISION_POINT>\n<LR_FACTORS>build, review</LR_FACTORS>\n<LR_BRANCH factor=\"build\" id=\"build\">...\nSTOP_POINT: ...\n</LR_BRANCH>\n<LR_EVALUATE_BRANCHES/>"}
```

Use:

```bash
PYTHONPATH=src python3 scripts/build_branching_sft_dataset.py
```

The seed dataset can be validated without CUDA:

```bash
python3 scripts/train_branching_lora.py --dry-run
```
