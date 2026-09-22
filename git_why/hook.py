from __future__ import annotations

import stat
from pathlib import Path

HOOK_SCRIPT = """#!/bin/sh
# installed by git-why
COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2

if [ -z "$COMMIT_SOURCE" ]; then
  git-why explain --staged --commit-message-only > "$COMMIT_MSG_FILE.git-why" 2>/dev/null
  if [ -s "$COMMIT_MSG_FILE.git-why" ]; then
    cat "$COMMIT_MSG_FILE" >> "$COMMIT_MSG_FILE.git-why"
    mv "$COMMIT_MSG_FILE.git-why" "$COMMIT_MSG_FILE"
  fi
fi
"""


def install_hook(repo_path: Path) -> Path:
    hooks_dir = repo_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "prepare-commit-msg"
    hook_path.write_text(HOOK_SCRIPT)
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC)
    return hook_path
