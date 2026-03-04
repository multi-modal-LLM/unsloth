"""
rl/reward.py

Reward function for GRPO training.

Binary reward signal:
  - 1.0 if the model's generated Pintora diagram renders without errors
  - 0.0 if the diagram has syntax errors or fails to render

The function signature follows the TRL GRPOTrainer reward_funcs convention:
    reward_fn(completions: list[str], **kwargs) -> list[float]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pintora_cli import render_diagram

# ---------------------------------------------------------------------------
# Diagram code extraction
# ---------------------------------------------------------------------------

# Pintora diagrams start with @start<type> or @startuml and end with @end<type> / @enduml
_PINTORA_BLOCK_RE = re.compile(
    r"(@start\w+.*?@end\w+)",
    re.DOTALL | re.IGNORECASE,
)


def extract_diagram_code(text: str) -> str | None:
    """
    Extract the first Pintora diagram block from a model completion.

    The model may wrap the code in markdown fences or produce preceding
    explanation text. This function strips those wrappings.
    """
    # Try extracting from markdown fences first
    fence_match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if re.search(r"@start\w+", candidate, re.IGNORECASE):
            return candidate

    # Fall back: find the raw @start…@end block
    block_match = _PINTORA_BLOCK_RE.search(text)
    if block_match:
        return block_match.group(1).strip()

    # Last resort: return the raw completion and let @pintora/cli decide
    return text.strip() if text.strip() else None


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def pintora_render_reward(completions: list[str], **kwargs) -> list[float]:
    """
    TRL-compatible reward function for GRPO training.

    Args:
        completions: List of model-generated strings (one per sample in the batch).
        **kwargs:    Ignored (TRL may pass prompts, inputs, etc.)

    Returns:
        List of float rewards in the same order as `completions`.
        1.0 = rendered successfully, 0.0 = failed.
    """
    rewards: list[float] = []
    for completion in completions:
        code = extract_diagram_code(completion)
        if code is None:
            rewards.append(0.0)
            continue
        success, _ = render_diagram(code)
        rewards.append(1.0 if success else 0.0)
    return rewards


def pintora_render_reward_verbose(
    completions: list[str],
    **kwargs,
) -> tuple[list[float], list[str]]:
    """
    Same as pintora_render_reward but also returns error messages for debugging.
    Not used by GRPOTrainer directly — call this manually when testing.
    """
    rewards: list[float] = []
    messages: list[str] = []
    for completion in completions:
        code = extract_diagram_code(completion)
        if code is None:
            rewards.append(0.0)
            messages.append("no diagram code found in completion")
            continue
        success, err = render_diagram(code)
        rewards.append(1.0 if success else 0.0)
        messages.append("" if success else err)
    return rewards, messages
