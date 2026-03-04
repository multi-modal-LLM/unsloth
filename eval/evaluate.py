"""
eval/evaluate.py

Evaluation script: generate 1000 randomized prompts, run inference with the
fine-tuned model, validate each output with @pintora/cli, and produce CSV results.

Matches the methodology from the blog post:
  - Random instruction built from entity/action/type pools
  - 1000 samples
  - Output: pintora_eval.csv, pintora_eval_good.csv, pintora_eval_bad.csv

Usage:
    python eval/evaluate.py --checkpoint outputs/ift-checkpoint
    python eval/evaluate.py --checkpoint outputs/rl-checkpoint --num-samples 500
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.prompts import ENTITIES, ACTIONS, EVAL_DIAGRAM_TYPES
from utils.pintora_cli import render_diagram

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Inference template (must match IFT training format)
# ---------------------------------------------------------------------------

INFERENCE_PROMPT = """\
Pintora Diagram Edit Instruction
### Instruction: {instruction}

### Response: """


# ---------------------------------------------------------------------------
# Random task generator (from the blog post)
# ---------------------------------------------------------------------------

def create_from_scratch_task() -> dict:
    """
    Build one random "generate from scratch" task.
    Returns {"instruction": str, "input": str} — no output (that's what the model generates).
    """
    d_type = random.choice(EVAL_DIAGRAM_TYPES)
    num_interactions = random.randint(1, 3)
    interactions = []
    for _ in range(num_interactions):
        src, dst = random.sample(ENTITIES, 2)
        action = random.choice(ACTIONS)
        interactions.append(f"{src} {action} to {dst}")
    prompt_desc = ", and then ".join(interactions)
    instruction = f"Create a {d_type} that shows: {prompt_desc}."
    return {"instruction": instruction, "input": ""}


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(checkpoint: str):
    """Load the fine-tuned model from an Unsloth checkpoint."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_diagram(model, tokenizer, instruction: str, max_new_tokens: int = 512) -> str:
    """Run inference and return the model's completion."""
    import torch

    prompt = INFERENCE_PROMPT.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Slice off the prompt tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Pintora fine-tuned model")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the fine-tuned model checkpoint (IFT or RL)",
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"[eval] Loading model from {args.checkpoint}…")
    model, tokenizer = load_model(args.checkpoint)

    all_rows: list[dict] = []
    good_rows: list[dict] = []
    bad_rows:  list[dict] = []

    print(f"[eval] Running {args.num_samples} random evaluations…")
    for i in tqdm(range(args.num_samples), desc="evaluate"):
        task = create_from_scratch_task()
        instruction = task["instruction"]
        output_code = generate_diagram(model, tokenizer, instruction, args.max_new_tokens)
        success, err = render_diagram(output_code)

        row = {
            "index": i,
            "instruction": instruction,
            "output": output_code,
            "success": success,
            "error": err,
        }
        all_rows.append(row)
        (good_rows if success else bad_rows).append(row)

    # Write CSVs
    fieldnames = ["index", "instruction", "output", "success", "error"]
    for name, rows in [
        ("pintora_eval.csv",      all_rows),
        ("pintora_eval_good.csv", good_rows),
        ("pintora_eval_bad.csv",  bad_rows),
    ]:
        path = RESULTS_DIR / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eval] Saved {len(rows)} rows → {path}")

    total = len(all_rows)
    accuracy = 100 * len(good_rows) / total if total else 0
    print(
        f"\n[eval] Results: {len(good_rows)}/{total} diagrams rendered successfully "
        f"({accuracy:.1f}% accuracy)"
    )


if __name__ == "__main__":
    main()
