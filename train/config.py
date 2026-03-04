"""
train/config.py

Shared training configuration dataclass for CPT and IFT phases.
Import this and override fields as needed before calling the train scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TrainingConfig:
    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    base_model: str = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    load_in_4bit: bool = True         # 4-bit QLoRA quantization

    # -----------------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------------
    lora_r: int = 64                  # LoRA rank
    lora_alpha: int = 64              # LoRA alpha (= r keeps scaling neutral)
    lora_dropout: float = 0.05
    # Modules to target with LoRA adapters.
    # NOTE: Intentionally excluding embed_tokens and lm_head for CPT to save ~5-6GB VRAM.
    # The IFT config adds them back via `ift_include_embedding_modules`.
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])

    # -----------------------------------------------------------------------
    # SFT / training loop
    # -----------------------------------------------------------------------
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch = 8
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "epoch"
    fp16: bool = False                # disabled; use bf16 on Ampere+ GPUs
    bf16: bool = True

    # -----------------------------------------------------------------------
    # Unsloth
    # -----------------------------------------------------------------------
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42

    # -----------------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------------
    cpt_output_dir: str = os.environ.get("CPT_OUTPUT_DIR", "outputs/cpt-checkpoint")
    ift_output_dir: str = os.environ.get("IFT_OUTPUT_DIR", "outputs/ift-checkpoint")
    rl_output_dir: str  = os.environ.get("RL_OUTPUT_DIR",  "outputs/rl-checkpoint")

    # -----------------------------------------------------------------------
    # Data paths
    # -----------------------------------------------------------------------
    cpt_dataset_path: str = "data/processed/cpt_dataset.jsonl"
    ift_dataset_path: str = "data/processed/ift_dataset.jsonl"
