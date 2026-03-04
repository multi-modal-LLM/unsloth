# Pintora Coder — Teaching an LLM a Niche Diagramming Language

## Goal
Fine-tune **Qwen2.5-Coder-7B** to generate and edit [Pintora](https://pintorajs.vercel.app/) diagrams
using a two-phase training pipeline, then optionally improve accuracy with RL (GRPO).

## Project Structure

```
pintora-llm/
├── data/
│   ├── pintora_docs.py       # Pintora syntax reference (used as LLM prompt context)
│   ├── generate_data.py      # Generate raw CPT + IFT rows via LLM (Gemini / Claude)
│   ├── validate_data.py      # Validate each diagram output via @pintora/cli
│   ├── deduplicate.py        # Merge rows with identical output fields
│   └── upload_hf.py          # Push final datasets to HuggingFace Hub
│
├── train/
│   ├── config.py             # TrainingConfig dataclass (all hyperparams)
│   ├── cpt_train.py          # Phase 1: Continued Pre-Training
│   └── ift_train.py          # Phase 2: Instruction Fine-Tuning
│
├── rl/
│   ├── reward.py             # Reward function: 1.0 if diagram renders, else 0.0
│   └── grpo_train.py         # GRPO-based RL training with trl.GRPOTrainer
│
├── eval/
│   ├── prompts.py            # Entity / action / diagram-type lists for random prompts
│   └── evaluate.py           # Run 1000 random prompts → validate → output CSV
│
├── utils/
│   ├── pintora_cli.py        # Python wrapper for `npx @pintora/cli render`
│   └── hf_dataset.py         # JSONL ↔ HuggingFace Dataset helpers
│
├── tests/
│   ├── test_pintora_cli.py
│   ├── test_reward.py
│   ├── test_deduplicate.py
│   └── test_generate_data.py
│
├── .env.example
└── requirements.txt
```

## Pipeline Overview

```
1. GENERATE   data/generate_data.py   → data/raw/cpt_raw.jsonl + ift_raw.jsonl
2. VALIDATE   data/validate_data.py   → data/processed/cpt_dataset.jsonl + ift_dataset.jsonl
3. CPT TRAIN  train/cpt_train.py      → outputs/cpt-checkpoint/
4. IFT TRAIN  train/ift_train.py      → outputs/ift-checkpoint/
5. RL TRAIN   rl/grpo_train.py        → outputs/rl-checkpoint/     (optional)
6. EVALUATE   eval/evaluate.py        → eval/results/*.csv
7. UPLOAD     data/upload_hf.py       → HuggingFace Hub
```

## Setup

### Prerequisites
- Python ≥ 3.10
- Node.js ≥ 18 (for `@pintora/cli`)
- CUDA GPU with ≥ 16 GB VRAM (A40 or A100 recommended for 7B QLoRA)

### Install Python deps
```bash
pip install -r requirements.txt
```

### Install Pintora CLI (for validation)
```bash
npm install -g @pintora/cli
```

### Configure API keys
```bash
cp .env.example .env
# Fill in GEMINI_API_KEY / ANTHROPIC_API_KEY / HF_TOKEN
```

## Data Generation

```bash
# Generate ~2000 rows using Gemini (default)
python data/generate_data.py --provider gemini --cpt-count 1000 --ift-count 1000

# Validate and clean
python data/validate_data.py

# Deduplicate
python data/deduplicate.py
```

## Training

```bash
# Phase 1: CPT
python train/cpt_train.py

# Phase 2: IFT (loads CPT adapter automatically)
python train/ift_train.py
```

## RL Fine-tuning (GRPO)

```bash
python rl/grpo_train.py --checkpoint outputs/ift-checkpoint
```

## Evaluation

```bash
python eval/evaluate.py --checkpoint outputs/ift-checkpoint --num-samples 1000
# Results written to eval/results/
```

## References
- [Pintora docs](https://pintorajs.vercel.app/docs/intro/)
- [Unsloth continued pretraining notebook](https://docs.unsloth.ai/basics/continued-pretraining)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Model on HuggingFace](https://huggingface.co/huytd189/pintora-coder-7b)
- [CPT Dataset](https://huggingface.co/datasets/huytd189/pintora-instruct)
- [IFT Dataset](https://huggingface.co/datasets/huytd189/pintora-edit-instruct)
