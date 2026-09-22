from __future__ import annotations

import subprocess
from pathlib import Path


def get_diff(repo_path: Path, ref: str) -> str:
    """Return the diff introduced by `ref` (a commit sha) against its first parent.

    For a range like "main..feature" or the literal "--staged", pass through to
    `git diff` / `git show` directly via `get_diff_range` / `get_staged_diff` instead.
    """
    result = subprocess.run(
        ["git", "show", "--no-color", "--pretty=format:", ref],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show failed for ref '{ref}': {result.stderr.strip()}")
    return result.stdout


def get_diff_range(repo_path: Path, range_spec: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-color", range_spec],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed for range '{range_spec}': {result.stderr.strip()}")
    return result.stdout


def get_staged_diff(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-color", "--cached"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff --cached failed: {result.stderr.strip()}")
    return result.stdout
