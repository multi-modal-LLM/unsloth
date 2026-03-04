"""
data/deduplicate.py

Deduplicate JSONL datasets by merging rows that share the same `output` value.
When two rows have the same output, the one with the longer / richer instruction is kept.

Usage:
    python data/deduplicate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hf_dataset import load_jsonl, save_jsonl

PROCESSED_DIR = Path("data/processed")
CPT_IN = PROCESSED_DIR / "cpt_dataset.jsonl"
IFT_IN = PROCESSED_DIR / "ift_dataset.jsonl"


def deduplicate(rows: list[dict]) -> list[dict]:
    """
    Merge rows that share the same `output` field.
    When there are duplicates, keep the one whose `instruction` is longest
    (heuristic for richest context).
    """
    seen: dict[str, dict] = {}
    for row in rows:
        key = row.get("output", "").strip()
        if not key:
            continue
        if key not in seen:
            seen[key] = row
        else:
            # Prefer whichever has a longer instruction
            existing_len = len(seen[key].get("instruction", ""))
            new_len = len(row.get("instruction", ""))
            if new_len > existing_len:
                seen[key] = row
    return list(seen.values())


def process_file(path: Path, label: str) -> None:
    if not path.exists():
        print(f"[skip] {label}: {path} not found")
        return
    rows = load_jsonl(path)
    before = len(rows)
    rows = deduplicate(rows)
    after = len(rows)
    save_jsonl(rows, path)
    print(f"[{label}] {before} → {after} rows  (removed {before - after} duplicates)")


def main() -> None:
    process_file(CPT_IN, "CPT")
    process_file(IFT_IN, "IFT")
    print("[done] Deduplication complete.")


if __name__ == "__main__":
    main()
