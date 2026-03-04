"""
data/upload_hf.py

Upload the cleaned datasets to HuggingFace Hub.

Usage:
    python data/upload_hf.py
    python data/upload_hf.py --cpt-repo myname/pintora-instruct --ift-repo myname/pintora-edit-instruct
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hf_dataset import push_jsonl_to_hub

load_dotenv()

PROCESSED_DIR = Path("data/processed")
CPT_PATH = PROCESSED_DIR / "cpt_dataset.jsonl"
IFT_PATH = PROCESSED_DIR / "ift_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Pintora datasets to HuggingFace Hub")
    parser.add_argument(
        "--cpt-repo",
        default=os.environ.get("HF_CPT_DATASET_REPO", ""),
        help="HuggingFace repo id for CPT dataset (e.g. myname/pintora-instruct)",
    )
    parser.add_argument(
        "--ift-repo",
        default=os.environ.get("HF_IFT_DATASET_REPO", ""),
        help="HuggingFace repo id for IFT dataset (e.g. myname/pintora-edit-instruct)",
    )
    parser.add_argument("--private", action="store_true", help="Create private datasets")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    if not args.cpt_repo or not args.ift_repo:
        print(
            "[error] Set --cpt-repo and --ift-repo, "
            "or HF_CPT_DATASET_REPO / HF_IFT_DATASET_REPO env vars."
        )
        sys.exit(1)

    if not CPT_PATH.exists():
        print(f"[error] CPT dataset not found: {CPT_PATH}. Run validate_data.py first.")
        sys.exit(1)
    if not IFT_PATH.exists():
        print(f"[error] IFT dataset not found: {IFT_PATH}. Run validate_data.py first.")
        sys.exit(1)

    print(f"Uploading CPT dataset to {args.cpt_repo}…")
    push_jsonl_to_hub(CPT_PATH, args.cpt_repo, token=token, private=args.private)

    print(f"Uploading IFT dataset to {args.ift_repo}…")
    push_jsonl_to_hub(IFT_PATH, args.ift_repo, token=token, private=args.private)

    print("[done] Datasets uploaded.")


if __name__ == "__main__":
    main()
