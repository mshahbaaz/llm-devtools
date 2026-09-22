from __future__ import annotations

import re
import subprocess
from pathlib import Path

_DEF_LINE_RE = re.compile(r"^\+?\s*def\s+(\w+)\s*\(")


def find_changed_functions(diff_text: str) -> list[str]:
    """Extract Python function names touched by a diff, from lines like
    '+def check_password(pw):' or unchanged context lines showing a def. This is
    intentionally simple (regex, not an AST); it only needs to be good enough to
    point the explanation prompt at relevant names, not to be a complete parser.
    """
    names = []
    for line in diff_text.splitlines():
        match = _DEF_LINE_RE.match(line)
        if match:
            names.append(match.group(1))
    return names


def find_test_references(repo_path: Path, function_name: str) -> list[str]:
    """Grep the repo for files that reference `function_name`, so the explanation can
    say whether tests exist for the changed function (a real risk signal) instead of
    guessing.
    """
    result = subprocess.run(
        ["git", "grep", "-l", function_name],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):  # 1 = no matches, not an error
        return []
    return [line for line in result.stdout.splitlines() if line]
