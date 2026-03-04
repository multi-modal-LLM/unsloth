"""
tests/test_reward.py

Unit tests for rl/reward.py using mocked render_diagram calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.reward import pintora_render_reward, extract_diagram_code


class TestExtractDiagramCode:
    def test_raw_diagram_block(self):
        code = "@startsequence\nA -> B: hello\n@endsequence"
        result = extract_diagram_code(code)
        assert result == code

    def test_markdown_fenced_code(self):
        text = "Here is the diagram:\n```pintora\n@startsequence\nA -> B: hi\n@endsequence\n```"
        result = extract_diagram_code(text)
        assert "@startsequence" in result

    def test_explanation_before_diagram(self):
        text = "Sure! Here you go:\n@startcomponent\ncomponent A\n@endcomponent"
        result = extract_diagram_code(text)
        assert "@startcomponent" in result

    def test_returns_none_for_no_diagram(self):
        result = extract_diagram_code("")
        assert result is None

    def test_plain_text_no_diagram_returns_stripped(self):
        # Non-empty text with no @start block returns the text itself
        text = "I cannot generate that diagram."
        result = extract_diagram_code(text)
        # Should not be None — falls through to last-resort strip
        assert result == text.strip()


class TestPintoraRenderReward:
    def _mock_render(self, results: list[tuple[bool, str]]):
        """Return a side_effect list for rendering multiple calls."""
        return results

    def test_all_success_returns_ones(self):
        completions = [
            "@startsequence\nA -> B: hi\n@endsequence",
            "@startcomponent\ncomponent C\n@endcomponent",
        ]
        with patch(
            "rl.reward.render_diagram",
            side_effect=[(True, ""), (True, "")],
        ):
            rewards = pintora_render_reward(completions)

        assert rewards == [1.0, 1.0]

    def test_all_failure_returns_zeros(self):
        completions = ["bad code 1", "bad code 2"]
        with patch(
            "rl.reward.render_diagram",
            side_effect=[(False, "err"), (False, "err")],
        ):
            rewards = pintora_render_reward(completions)

        assert rewards == [0.0, 0.0]

    def test_mixed_results(self):
        completions = [
            "@startsequence\nA -> B: ok\n@endsequence",   # success
            "invalid diagram",                              # fail
            "@startcomponent\ncomponent X\n@endcomponent", # success
        ]
        with patch(
            "rl.reward.render_diagram",
            side_effect=[(True, ""), (False, "parse error"), (True, "")],
        ):
            rewards = pintora_render_reward(completions)

        assert rewards == [1.0, 0.0, 1.0]

    def test_empty_completion_scores_zero(self):
        rewards = pintora_render_reward([""])
        assert rewards == [0.0]

    def test_return_length_matches_input(self):
        completions = ["@startsequence\nA->B: x\n@endsequence"] * 8
        with patch("rl.reward.render_diagram", return_value=(True, "")):
            rewards = pintora_render_reward(completions)
        assert len(rewards) == 8

    def test_extra_kwargs_ignored(self):
        """GRPOTrainer may pass extra kwargs — they should be ignored."""
        completions = ["@startsequence\nA->B: hi\n@endsequence"]
        with patch("rl.reward.render_diagram", return_value=(True, "")):
            rewards = pintora_render_reward(
                completions,
                prompts=["some prompt"],
                inputs=["some input"],
            )
        assert rewards == [1.0]
