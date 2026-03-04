"""
data/validate_data.py

Validate raw generated data by rendering each diagram with @pintora/cli.
Removes rows that fail to render, then writes cleaned datasets to data/processed/.

Usage:
    python data/validate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hf_dataset import load_jsonl, save_jsonl
from utils.pintora_cli import render_diagram

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CPT_RAW = RAW_DIR / "cpt_raw.jsonl"
IFT_RAW = RAW_DIR / "ift_raw.jsonl"
CPT_OUT = PROCESSED_DIR / "cpt_dataset.jsonl"
IFT_OUT = PROCESSED_DIR / "ift_dataset.jsonl"


def validate_dataset(input_path: Path, output_path: Path, label: str) -> None:
    rows = load_jsonl(input_path)
    print(f"\n[{label}] Validating {len(rows)} rows…")

    valid: list[dict] = []
    invalid: list[dict] = []

    for row in tqdm(rows, desc=label):
        code = row.get("output", "").strip()
        if not code:
            invalid.append(row)
            continue
        success, err = render_diagram(code)
        if success:
            valid.append(row)
        else:
            invalid.append({**row, "_error": err})

    save_jsonl(valid, output_path)
    error_path = output_path.with_name(output_path.stem + "_errors.jsonl")
    save_jsonl(invalid, error_path)

    total = len(rows)
    pct = 100 * len(valid) / total if total else 0
    print(
        f"[{label}] Valid: {len(valid)}/{total} ({pct:.1f}%)  "
        f"Errors saved to: {error_path}"
    )


def main() -> None:
    if not CPT_RAW.exists():
        print(f"[error] CPT raw file not found: {CPT_RAW}. Run generate_data.py first.")
        sys.exit(1)
    if not IFT_RAW.exists():
        print(f"[error] IFT raw file not found: {IFT_RAW}. Run generate_data.py first.")
        sys.exit(1)

    validate_dataset(CPT_RAW, CPT_OUT, "CPT")
    validate_dataset(IFT_RAW, IFT_OUT, "IFT")
    print("\n[done] Validation complete.")


if __name__ == "__main__":
    main()
