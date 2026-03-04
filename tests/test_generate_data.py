"""
tests/test_generate_data.py

Unit tests for data/generate_data.py — mock LLM calls to verify
prompt building, JSON parsing, and row normalisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_data import (
    _parse_json_array,
    generate_cpt_batch,
    generate_ift_batch,
)


class TestParseJsonArray:
    def test_plain_json(self):
        raw = '[{"output": "code1"}, {"output": "code2"}]'
        result = _parse_json_array(raw)
        assert len(result) == 2
        assert result[0]["output"] == "code1"

    def test_strips_markdown_fences(self):
        raw = '```json\n[{"output": "fenced"}]\n```'
        result = _parse_json_array(raw)
        assert result[0]["output"] == "fenced"

    def test_strips_bare_code_fence(self):
        raw = '```\n[{"output": "bare"}]\n```'
        result = _parse_json_array(raw)
        assert result[0]["output"] == "bare"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_array("not json at all")

    def test_empty_array(self):
        result = _parse_json_array("[]")
        assert result == []


class TestGenerateCptBatch:
    def _make_llm(self, rows: list[dict]):
        """Return a callable that returns the JSON rows as a string."""
        def _call(prompt: str) -> str:
            return json.dumps(rows)
        return _call

    def test_returns_normalised_cpt_rows(self):
        fake_rows = [{"output": "@startsequence\nA->B\n@endsequence"}] * 5
        call_llm = self._make_llm(fake_rows)
        result = generate_cpt_batch(call_llm, batch_size=5)
        assert len(result) == 5
        for row in result:
            assert "instruction" in row
            assert "input" in row
            assert "output" in row
            assert row["instruction"] == ""
            assert row["input"] == ""

    def test_empty_outputs_preserved(self):
        """generate_cpt_batch should NOT filter here — that's done downstream."""
        fake_rows = [{"output": ""}]
        call_llm = self._make_llm(fake_rows)
        result = generate_cpt_batch(call_llm, batch_size=1)
        assert len(result) == 1
        assert result[0]["output"] == ""


class TestGenerateIftBatch:
    def _make_llm(self, rows: list[dict]):
        def _call(prompt: str) -> str:
            return json.dumps(rows)
        return _call

    def test_returns_normalised_ift_rows(self):
        fake_rows = [
            {"instruction": "Make a seq diagram", "input": "", "output": "@startsequence\n@endsequence"},
            {"instruction": "Edit this", "input": "@startseq...", "output": "@startsequence\n@endsequence"},
        ]
        call_llm = self._make_llm(fake_rows)
        result = generate_ift_batch(call_llm, batch_size=2)
        assert len(result) == 2
        for row in result:
            assert "instruction" in row
            assert "input" in row
            assert "output" in row

    def test_missing_fields_default_to_empty_string(self):
        """LLM might return partial rows — should not raise."""
        fake_rows = [{"output": "@startsequence\n@endsequence"}]
        call_llm = self._make_llm(fake_rows)
        result = generate_ift_batch(call_llm, batch_size=1)
        assert result[0]["instruction"] == ""
        assert result[0]["input"] == ""
