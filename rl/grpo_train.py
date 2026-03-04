"""
rl/grpo_train.py

Reinforcement Learning via GRPO (Group Relative Policy Optimization).

Uses TRL's GRPOTrainer with a binary reward: 1.0 if the generated Pintora
diagram renders without errors (via @pintora/cli), 0.0 otherwise.

This addresses the blog post's "next step" of using RL to improve syntactic
accuracy beyond what CPT + IFT achieves.

Usage:
    python rl/grpo_train.py --checkpoint outputs/ift-checkpoint
    python rl/grpo_train.py --checkpoint outputs/ift-checkpoint --num-steps 500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.config import TrainingConfig
from utils.hf_dataset import load_jsonl
from rl.reward import pintora_render_reward

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt template (same as IFT — model already knows this format)
# ---------------------------------------------------------------------------

INFERENCE_PROMPT = """\
Pintora Diagram Edit Instruction
### Instruction: {instruction}
{input}
### Response: """


def format_prompt(row: dict) -> str:
    inp = row.get("input", "").strip()
    input_block = f"### Input Diagram:\n{inp}" if inp else ""
    return INFERENCE_PROMPT.format(
        instruction=row.get("instruction", "").strip(),
        input=input_block,
    )


# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO RL training for Pintora LLM")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("IFT_OUTPUT_DIR", "outputs/ift-checkpoint"),
        help="Path to the IFT fine-tuned model checkpoint",
    )
    parser.add_argument("--num-steps",    type=int,   default=200,  help="Total GRPO steps")
    parser.add_argument("--batch-size",   type=int,   default=4,    help="Per-device batch size")
    parser.add_argument("--num-gen",      type=int,   default=4,    help="Completions per prompt (G)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir",   default=None)
    args = parser.parse_args()

    cfg = TrainingConfig()
    output_dir = args.output_dir or cfg.rl_output_dir

    # Late imports — only when actually running
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    print(f"[RL] Loading IFT checkpoint: {args.checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.lora_target_modules + ["embed_tokens", "lm_head"],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
    )

    # Load IFT dataset as RL prompts (instruction-only, no gold output needed)
    print(f"[RL] Loading prompts from: {cfg.ift_dataset_path}")
    rows = load_jsonl(cfg.ift_dataset_path)
    prompts = [{"prompt": format_prompt(r)} for r in rows if r.get("instruction", "").strip()]
    dataset = Dataset.from_list(prompts)
    print(f"[RL] Prompt dataset size: {len(dataset)}")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=5e-6,              # lower LR for RL fine-tuning
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        seed=cfg.random_state,
        # GRPO-specific
        num_generations=args.num_gen,    # G = number of completions per prompt
        max_new_tokens=args.max_new_tokens,
        temperature=0.9,
        top_p=0.95,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[pintora_render_reward],
        args=grpo_config,
        train_dataset=dataset,
    )

    print("[RL] Starting GRPO training…")
    trainer.train()

    print(f"[RL] Saving RL adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[RL] Done.")


if __name__ == "__main__":
    main()
