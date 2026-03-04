"""
utils/pintora_cli.py

Python wrapper around the `@pintora/cli` Node.js command-line tool.
Uses subprocess to render a Pintora diagram code string, and reports
whether the render succeeded or failed.

Prerequisites:
    npm install -g @pintora/cli
    (or use npx @pintora/cli)
"""

from __future__ import annotations

import subprocess
import tempfile
import os
from pathlib import Path


# Path to the pintora CLI — override via PINTORA_CLI_PATH env var.
# Defaults to using npx so no global install is required.
PINTORA_CLI = os.environ.get("PINTORA_CLI_PATH", "npx")
PINTORA_CLI_ARGS = (
    ["@pintora/cli", "render"]
    if PINTORA_CLI == "npx"
    else ["render"]
)

# Timeout for a single render call (seconds)
RENDER_TIMEOUT = int(os.environ.get("PINTORA_RENDER_TIMEOUT", "15"))


def render_diagram(code: str) -> tuple[bool, str]:
    """
    Attempt to render a Pintora diagram code string.

    The function writes the code to a temp file and passes its path to
    @pintora/cli. Writing to a file avoids shell injection issues and
    handles multi-line code blocks correctly on Windows / Unix.

    Args:
        code: Pintora diagram source code (e.g. @startsequence ... @endsequence)

    Returns:
        (success: bool, message: str)
        - success=True  → diagram rendered without errors
        - success=False → message contains the stderr output from @pintora/cli
    """
    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pintora",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [PINTORA_CLI] + PINTORA_CLI_ARGS + ["--input", tmp_path],
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT,
        )
        if result.returncode == 0:
            return True, ""
        else:
            err = (result.stderr or result.stdout or "unknown error").strip()
            return False, err
    except subprocess.TimeoutExpired:
        return False, f"render timed out after {RENDER_TIMEOUT}s"
    except FileNotFoundError:
        return False, (
            "pintora CLI not found. "
            "Run: npm install -g @pintora/cli  or set PINTORA_CLI_PATH env var."
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def batch_render(codes: list[str]) -> list[tuple[bool, str]]:
    """
    Render a list of Pintora code strings sequentially.

    Returns a list of (success, message) tuples in the same order.
    """
    return [render_diagram(code) for code in codes]
