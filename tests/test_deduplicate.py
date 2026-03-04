"""
tests/test_deduplicate.py

Unit tests for data/deduplicate.py deduplication logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.deduplicate import deduplicate


class TestDeduplicate:
    def test_no_duplicates_unchanged(self):
        rows = [
            {"instruction": "make A", "input": "", "output": "@startsequence\nA->B\n@endsequence"},
            {"instruction": "make B", "input": "", "output": "@startcomponent\ncomponent X\n@endcomponent"},
        ]
        result = deduplicate(rows)
        assert len(result) == 2

    def test_duplicate_output_merged_to_one(self):
        same_output = "@startsequence\nA->B: hi\n@endsequence"
        rows = [
            {"instruction": "short", "input": "", "output": same_output},
            {"instruction": "short", "input": "", "output": same_output},
        ]
        result = deduplicate(rows)
        assert len(result) == 1

    def test_richer_instruction_kept(self):
        same_output = "@startsequence\nA->B: hi\n@endsequence"
        rows = [
            {"instruction": "short",               "input": "", "output": same_output},
            {"instruction": "a much longer instruction that is richer", "input": "", "output": same_output},
        ]
        result = deduplicate(rows)
        assert len(result) == 1
        assert result[0]["instruction"] == "a much longer instruction that is richer"

    def test_empty_output_rows_skipped(self):
        rows = [
            {"instruction": "x", "input": "", "output": ""},
            {"instruction": "y", "input": "", "output": "   "},
        ]
        result = deduplicate(rows)
        assert len(result) == 0

    def test_multiple_groups_deduplicated_independently(self):
        rows = [
            {"instruction": "a", "input": "", "output": "diag1"},
            {"instruction": "longer a", "input": "", "output": "diag1"},
            {"instruction": "b", "input": "", "output": "diag2"},
            {"instruction": "c", "input": "", "output": "diag3"},
            {"instruction": "longer c", "input": "", "output": "diag3"},
        ]
        result = deduplicate(rows)
        assert len(result) == 3
        instrs = {r["output"]: r["instruction"] for r in result}
        assert instrs["diag1"] == "longer a"
        assert instrs["diag3"] == "longer c"

    def test_empty_input_returns_empty(self):
        assert deduplicate([]) == []
