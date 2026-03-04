"""
train/ift_train.py

Phase 2: Instruction Fine-Tuning (IFT)

Loads the LoRA adapter produced by CPT, then fine-tunes on instruction-following
data so the model learns to generate/edit Pintora diagrams from natural-language
descriptions.

Prompt template (edit_prompt):
    Pintora Diagram Edit Instruction
    ### Instruction: {instruction}
    {input}
    ### Response: {output}

Usage:
    python train/ift_train.py
    python train/ift_train.py --cpt-checkpoint outputs/cpt-checkpoint --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.config import TrainingConfig
from utils.hf_dataset import load_jsonl

# ---------------------------------------------------------------------------
# Prompt template — matches the blog post exactly
# ---------------------------------------------------------------------------

EDIT_PROMPT = """\
Pintora Diagram Edit Instruction
### Instruction: {instruction}
{input}
### Response: {output}"""

# Inference-time prompt (no output, for generation)
INFERENCE_PROMPT = """\
Pintora Diagram Edit Instruction
### Instruction: {instruction}
{input}
### Response: """


def format_row(row: dict, eos_token: str) -> str:
    """Format a dataset row using the edit_prompt template."""
    inp = row.get("input", "").strip()
    input_block = f"### Input Diagram:\n{inp}" if inp else ""
    return (
        EDIT_PROMPT.format(
            instruction=row.get("instruction", "").strip(),
            input=input_block,
            output=row.get("output", "").strip(),
        )
        + eos_token
    )


def main(cfg: TrainingConfig | None = None, cpt_checkpoint: str | None = None) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    if cfg is None:
        cfg = TrainingConfig()

    # Load from CPT checkpoint if provided, otherwise from base model
    model_path = cpt_checkpoint or cfg.cpt_output_dir
    print(f"[IFT] Loading model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    # For IFT we add embed_tokens and lm_head to LoRA targets
    ift_targets = cfg.lora_target_modules + ["embed_tokens", "lm_head"]
    print("[IFT] Attaching LoRA adapters (including embed_tokens, lm_head)…")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=ift_targets,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
    )

    print(f"[IFT] Loading IFT dataset: {cfg.ift_dataset_path}")
    rows = load_jsonl(cfg.ift_dataset_path)
    eos = tokenizer.eos_token or "</s>"
    texts = [format_row(r, eos) for r in rows if r.get("instruction", "").strip()]
    dataset = Dataset.from_dict({"text": texts})
    print(f"[IFT] Dataset size: {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=cfg.ift_output_dir,
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

    print("[IFT] Starting training…")
    trainer.train()

    print(f"[IFT] Saving adapter to {cfg.ift_output_dir}")
    model.save_pretrained(cfg.ift_output_dir)
    tokenizer.save_pretrained(cfg.ift_output_dir)
    print("[IFT] Done.")


def parse_args() -> tuple[TrainingConfig, str | None]:
    parser = argparse.ArgumentParser(description="IFT training for Pintora LLM")
    parser.add_argument("--model",           default=None, help="Override base model / checkpoint")
    parser.add_argument("--cpt-checkpoint",  default=None, help="Path to CPT adapter checkpoint")
    parser.add_argument("--epochs",          type=int, default=None)
    parser.add_argument("--batch-size",      type=int, default=None)
    parser.add_argument("--output-dir",      default=None)
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.model:      cfg.base_model = args.model
    if args.epochs:     cfg.num_train_epochs = args.epochs
    if args.batch_size: cfg.per_device_train_batch_size = args.batch_size
    if args.output_dir: cfg.ift_output_dir = args.output_dir
    return cfg, args.cpt_checkpoint


if __name__ == "__main__":
    cfg, cpt_ckpt = parse_args()
    main(cfg, cpt_ckpt)
