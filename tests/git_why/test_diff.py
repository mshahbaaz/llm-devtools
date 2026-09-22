import subprocess
from pathlib import Path

from git_why.diff import get_diff


def test_get_diff_for_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add a.txt"], cwd=repo, check=True)

    diff = get_diff(repo, "HEAD")
    assert "a.txt" in diff
    assert "+hello" in diff


def test_get_diff_raises_on_bad_ref(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    try:
        get_diff(repo, "not-a-real-ref")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "not-a-real-ref" in str(e)
