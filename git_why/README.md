# git-why

Turn a diff or commit into a plain-English explanation of what changed and why,
grounded in whether the changed function actually has test coverage, not a guess.

## Usage

```bash
git-why explain HEAD                  # explain the last commit
git-why explain abc123                # explain a specific commit
git-why explain main..feature-branch  # explain a range
git-why explain --staged              # explain staged changes
git-why explain HEAD --commit-message-only  # print only the suggested commit message
git-why install-hook                  # auto-draft commit messages via a prepare-commit-msg hook
```

## How the risk assessment works

Before calling the model, `git-why` extracts the Python function names touched by
the diff and greps the repo for files that reference them (`git_why/context.py`).
That's fed into the prompt, so the RISK line reflects real test coverage instead
of the model guessing from the diff alone.
