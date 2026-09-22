import subprocess
from pathlib import Path

from git_why.context import find_changed_functions, find_test_references


def _make_repo_with_function_change(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)

    (repo / "auth.py").write_text("def check_password(pw):\n    return len(pw) > 0\n")
    subprocess.run(["git", "add", "auth.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)

    (repo / "auth.py").write_text("def check_password(pw):\n    return len(pw) >= 8\n")
    subprocess.run(["git", "add", "auth.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "require longer password"], cwd=repo, check=True)

    (repo / "test_auth.py").write_text("from auth import check_password\n\ndef test_check_password():\n    assert check_password('12345678')\n")
    subprocess.run(["git", "add", "test_auth.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add test"], cwd=repo, check=True)

    return repo


def test_find_changed_functions_extracts_python_def_names(tmp_path: Path):
    repo = _make_repo_with_function_change(tmp_path)
    diff_text = subprocess.run(
        ["git", "show", "--no-color", "--pretty=format:", "HEAD~1"], cwd=repo, capture_output=True, text=True
    ).stdout

    functions = find_changed_functions(diff_text)

    assert "check_password" in functions


def test_find_test_references_locates_files_mentioning_a_function(tmp_path: Path):
    repo = _make_repo_with_function_change(tmp_path)

    references = find_test_references(repo, function_name="check_password")

    assert any("test_auth.py" in ref for ref in references)
