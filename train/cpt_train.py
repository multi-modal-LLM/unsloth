"""
train/cpt_train.py

Phase 1: Continued Pre-Training (CPT)

Feed the model raw Pintora diagram code so it learns the language's syntax,
grammar, and token patterns — before any instruction-following is introduced.

Training setup:
  - Base model   : Qwen2.5-Coder-7B-Instruct (or override via .env BASE_MODEL)
  - Quantization : 4-bit QLoRA via Unsloth
  - LoRA targets : attention + MLP projections only (NOT embed_tokens / lm_head)
                   → saves ~5-6 GB VRAM vs including embedding layers
  - Format       : raw diagram code only (no instruction prefix)

Usage:
    python train/cpt_train.py
    python train/cpt_train.py --epochs 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.config import TrainingConfig
from utils.hf_dataset import load_jsonl


def build_cpt_text(row: dict) -> str:
    """
    For CPT we only care about the raw diagram code.
    Wrap it with the EOS token so the model learns sequence boundaries.
    """
    return row["output"].strip()


def main(cfg: TrainingConfig | None = None) -> None:
    # Inline import so the file is importable without GPU / unsloth installed
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    if cfg is None:
        cfg = TrainingConfig()

    print(f"[CPT] Loading base model: {cfg.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,             # auto-detect
        load_in_4bit=cfg.load_in_4bit,
    )

    print("[CPT] Attaching LoRA adapters (excluding embed_tokens / lm_head)…")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.lora_target_modules,   # no embed_tokens / lm_head
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
    )

    print(f"[CPT] Loading CPT dataset: {cfg.cpt_dataset_path}")
    rows = load_jsonl(cfg.cpt_dataset_path)
    texts = [build_cpt_text(r) for r in rows if r.get("output", "").strip()]
    dataset = Dataset.from_dict({"text": texts})
    print(f"[CPT] Dataset size: {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=cfg.cpt_output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        report_to="none",
        seed=cfg.random_state,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
    )

    print("[CPT] Starting training…")
    trainer.train()

    print(f"[CPT] Saving adapter to {cfg.cpt_output_dir}")
    model.save_pretrained(cfg.cpt_output_dir)
    tokenizer.save_pretrained(cfg.cpt_output_dir)
    print("[CPT] Done.")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="CPT training for Pintora LLM")
    parser.add_argument("--model",       default=None, help="Override base model")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--lora-r",      type=int,   default=None)
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.model:       cfg.base_model = args.model
    if args.epochs:      cfg.num_train_epochs = args.epochs
    if args.batch_size:  cfg.per_device_train_batch_size = args.batch_size
    if args.lora_r:      cfg.lora_r = args.lora_r
    if args.output_dir:  cfg.cpt_output_dir = args.output_dir
    return cfg


if __name__ == "__main__":
    main(parse_args())
