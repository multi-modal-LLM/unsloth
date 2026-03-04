"""
data/generate_data.py

Generate synthetic Pintora training data using an LLM (Gemini or Claude).
Each generated row has the form:
    { "instruction": str, "input": str, "output": str }

For CPT (Continued Pre-Training): instruction and input are empty; output is raw Pintora code.
For IFT (Instruction Fine-Tuning): instruction is a natural-language task; output is the diagram.

Usage:
    python data/generate_data.py --provider gemini --cpt-count 1000 --ift-count 1000
    python data/generate_data.py --provider claude  --cpt-count 500  --ift-count 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.pintora_docs import ALL_DOCS, DIAGRAM_TYPE_TO_DOCS, DIAGRAM_TYPES
from utils.hf_dataset import save_jsonl

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CPT_RAW_PATH = RAW_DIR / "cpt_raw.jsonl"
IFT_RAW_PATH = RAW_DIR / "ift_raw.jsonl"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CPT_SYSTEM_PROMPT = """You are an expert in the Pintora diagramming language.
Below is the complete Pintora syntax documentation:

{docs}

Generate {count} unique, syntactically correct Pintora diagrams.
Rules:
- Cover a variety of diagram types: sequenceDiagram, componentDiagram, activityDiagram,
  erDiagram, mindmap, ganttDiagram, classDiagram.
- Each diagram should be realistic and non-trivial (at least 6-10 elements).
- Output ONLY a JSON array. Each element is a dict with a single key "output" whose
  value is the full diagram code string. No markdown fences, no explanation.

Example format:
[
  {{"output": "@startsequence\\n...\\n@endsequence"}},
  ...
]
"""

IFT_SYSTEM_PROMPT = """You are an expert in the Pintora diagramming language.
Below is the complete Pintora syntax documentation:

{docs}

Generate {count} instruction-following training examples for Pintora diagram tasks.
Each example is one of two types:
  TYPE A — Generate from scratch:
    - "instruction": natural-language description of what diagram to create
    - "input": ""  (empty string)
    - "output": the complete Pintora diagram code

  TYPE B — Edit an existing diagram:
    - "instruction": natural-language description of what to change
    - "input": an existing Pintora diagram code to edit
    - "output": the edited version of the diagram

Rules:
- Mix TYPE A and TYPE B roughly 50/50.
- Cover all diagram types.
- Keep instructions realistic (things a developer might actually ask).
- Output ONLY a JSON array of objects with keys: instruction, input, output.
  No markdown, no explanation outside the JSON.
"""

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client = genai.GenerativeModel(model)
    response = client.generate_content(prompt)
    return response.text


def _call_claude(prompt: str, model: str = "claude-sonnet-4-5") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


PROVIDERS = {
    "gemini": _call_gemini,
    "claude": _call_claude,
}

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _parse_json_array(text: str) -> list[dict]:
    """Parse an LLM response that should contain a JSON array."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return json.loads(text)


def generate_cpt_batch(
    call_llm,
    batch_size: int = 50,
    diagram_type: str | None = None,
) -> list[dict]:
    """
    Ask the LLM to generate `batch_size` CPT rows.
    Returns a list of {"output": <pintora_code>} dicts.
    """
    docs = (
        DIAGRAM_TYPE_TO_DOCS.get(diagram_type, ALL_DOCS)
        if diagram_type
        else ALL_DOCS
    )
    prompt = CPT_SYSTEM_PROMPT.format(docs=docs, count=batch_size)
    raw = call_llm(prompt)
    rows = _parse_json_array(raw)
    # Normalise: CPT rows only need the output field
    return [{"instruction": "", "input": "", "output": r.get("output", "")} for r in rows]


def generate_ift_batch(
    call_llm,
    batch_size: int = 50,
) -> list[dict]:
    """
    Ask the LLM to generate `batch_size` IFT rows.
    Returns a list of {"instruction": .., "input": .., "output": ..} dicts.
    """
    prompt = IFT_SYSTEM_PROMPT.format(docs=ALL_DOCS, count=batch_size)
    raw = call_llm(prompt)
    rows = _parse_json_array(raw)
    return [
        {
            "instruction": r.get("instruction", ""),
            "input": r.get("input", ""),
            "output": r.get("output", ""),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def generate_dataset(
    mode: str,  # "cpt" or "ift"
    total: int,
    call_llm,
    output_path: Path,
    batch_size: int = 50,
    delay: float = 2.0,
) -> None:
    """
    Incrementally generate `total` rows and append to `output_path` (JSONL).
    Progress is saved after each batch so the run can be resumed.
    """
    existing = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        print(f"[resume] Found {existing} existing rows in {output_path}")

    needed = total - existing
    if needed <= 0:
        print(f"[skip] Already have {existing}/{total} rows for {mode}")
        return

    print(f"[generate] Generating {needed} more rows for {mode} (batch={batch_size})")
    diagram_types_cycle = DIAGRAM_TYPES.copy()

    generated = 0
    while generated < needed:
        batch_n = min(batch_size, needed - generated)
        try:
            if mode == "cpt":
                dtype = diagram_types_cycle[generated % len(diagram_types_cycle)]
                batch = generate_cpt_batch(call_llm, batch_size=batch_n, diagram_type=dtype)
            else:
                batch = generate_ift_batch(call_llm, batch_size=batch_n)

            # Filter empty outputs
            batch = [r for r in batch if r.get("output", "").strip()]
            save_jsonl(batch, output_path, append=True)
            generated += len(batch)
            print(f"  [{mode}] {existing + generated}/{total} rows saved")
        except Exception as exc:
            print(f"  [error] Batch failed: {exc}. Retrying in 5s…")
            time.sleep(5)
            continue

        if generated < needed:
            time.sleep(delay)

    print(f"[done] {mode.upper()} dataset: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pintora training data via LLM")
    parser.add_argument(
        "--provider",
        choices=["gemini", "claude"],
        default="gemini",
        help="LLM provider to use for generation",
    )
    parser.add_argument("--cpt-count", type=int, default=1000, help="Number of CPT rows")
    parser.add_argument("--ift-count", type=int, default=1000, help="Number of IFT rows")
    parser.add_argument("--batch-size", type=int, default=50, help="Rows per LLM call")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between batches")
    args = parser.parse_args()

    call_llm = PROVIDERS[args.provider]

    generate_dataset("cpt", args.cpt_count, call_llm, CPT_RAW_PATH, args.batch_size, args.delay)
    generate_dataset("ift", args.ift_count, call_llm, IFT_RAW_PATH, args.batch_size, args.delay)


if __name__ == "__main__":
    main()
