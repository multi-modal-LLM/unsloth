"""
utils/hf_dataset.py

HuggingFace dataset helpers:
- Load/save datasets in the project's JSONL format
- Convert between JSONL dicts and HuggingFace Dataset objects
- Push datasets to the Hub
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from datasets import Dataset
from huggingface_hub import HfApi


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[dict], path: str | Path, append: bool = False) -> None:
    """Save rows to a JSONL file. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def jsonl_to_dataset(path: str | Path) -> Dataset:
    """Load a JSONL file and return a HuggingFace Dataset."""
    rows = load_jsonl(path)
    return Dataset.from_list(rows)


def dataset_to_jsonl(dataset: Dataset, path: str | Path) -> None:
    """Save a HuggingFace Dataset to a JSONL file."""
    save_jsonl(dataset.to_list(), path)


# ---------------------------------------------------------------------------
# HuggingFace Hub
# ---------------------------------------------------------------------------

def push_dataset_to_hub(
    dataset: Dataset,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    """
    Push a HuggingFace Dataset to the Hub.

    Args:
        dataset:  The dataset to push.
        repo_id:  HuggingFace repo in the form "username/dataset-name".
        token:    HuggingFace access token (reads from HF_TOKEN env var if None).
        private:  Whether to create a private repository.
    """
    import os

    token = token or os.environ.get("HF_TOKEN")
    dataset.push_to_hub(repo_id, token=token, private=private)
    print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")


def push_jsonl_to_hub(
    jsonl_path: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    """Convenience: load a JSONL file and push it as a HuggingFace dataset."""
    dataset = jsonl_to_dataset(jsonl_path)
    push_dataset_to_hub(dataset, repo_id, token=token, private=private)
