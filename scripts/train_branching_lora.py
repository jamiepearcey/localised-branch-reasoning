#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUDA LoRA/QLoRA SFT for internalizing the branching protocol."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=Path, default=Path("data/branching_sft_seed.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("adapters/branching-lora"))
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action="store_true", help="Load the base model in 4-bit NF4.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training where supported.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and print config only.")
    args = parser.parse_args()

    examples = _load_jsonl(args.dataset)
    if not examples:
        raise SystemExit(f"dataset has no examples: {args.dataset}")
    for index, example in enumerate(examples):
        if not isinstance(example.get("text"), str) or not example["text"].strip():
            raise SystemExit(f"example {index} is missing non-empty text")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "dataset": str(args.dataset),
                    "examples": len(examples),
                    "output_dir": str(args.output_dir),
                    "qlora": args.qlora,
                    "max_seq_length": args.max_seq_length,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                },
                indent=2,
            )
        )
        return

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this training script")

    quantization_config = None
    model_kwargs = {"device_map": "auto"}
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    dataset = load_dataset("json", data_files=str(args.dataset), split="train")

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=5,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=not args.bf16,
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"adapter_output={args.output_dir}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"dataset does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON on line {line_number}: {exc}") from exc
    return rows


if __name__ == "__main__":
    main()
