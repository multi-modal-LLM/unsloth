"""
tests/test_pintora_cli.py

Unit tests for utils/pintora_cli.py using mocked subprocess calls.
No Node.js or @pintora/cli required to run these tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pintora_cli import render_diagram, batch_render


class TestRenderDiagram:
    VALID_CODE = "@startsequence\nA -> B: hello\n@endsequence"
    INVALID_CODE = "@startsequence\nBAD SYNTAX !!!\n@endsequence"

    def test_success_returns_true_and_empty_message(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("utils.pintora_cli.subprocess.run", return_value=mock_result):
            success, msg = render_diagram(self.VALID_CODE)

        assert success is True
        assert msg == ""

    def test_failure_returns_false_and_error_message(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ParseError: unexpected token BAD"

        with patch("utils.pintora_cli.subprocess.run", return_value=mock_result):
            success, msg = render_diagram(self.INVALID_CODE)

        assert success is False
        assert "ParseError" in msg

    def test_timeout_returns_false(self):
        with patch(
            "utils.pintora_cli.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="npx", timeout=15),
        ):
            success, msg = render_diagram(self.VALID_CODE)

        assert success is False
        assert "timed out" in msg

    def test_file_not_found_returns_false(self):
        with patch(
            "utils.pintora_cli.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            success, msg = render_diagram(self.VALID_CODE)

        assert success is False
        assert "not found" in msg.lower()

    def test_empty_code_still_calls_cli(self):
        """Empty code should still be passed to the CLI; the CLI decides if it's valid."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "empty input"
        mock_result.stdout = ""

        with patch("utils.pintora_cli.subprocess.run", return_value=mock_result):
            success, msg = render_diagram("")

        assert success is False


class TestBatchRender:
    def test_batch_returns_correct_length(self):
        codes = ["code1", "code2", "code3"]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        with patch("utils.pintora_cli.subprocess.run", return_value=mock_result):
            results = batch_render(codes)

        assert len(results) == 3
        assert all(success for success, _ in results)

    def test_batch_preserves_order(self):
        """Results must be in the same order as inputs."""
        return_values = [
            MagicMock(returncode=0, stderr="", stdout=""),
            MagicMock(returncode=1, stderr="err", stdout=""),
            MagicMock(returncode=0, stderr="", stdout=""),
        ]
        with patch("utils.pintora_cli.subprocess.run", side_effect=return_values):
            results = batch_render(["a", "b", "c"])

        assert results[0][0] is True
        assert results[1][0] is False
        assert results[2][0] is True
