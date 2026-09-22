import subprocess
from pathlib import Path

from git_why.hook import install_hook

HOOK_MARKER = "# installed by git-why"


def test_install_hook_writes_executable_script(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    hook_path = install_hook(repo)

    assert hook_path.exists()
    content = hook_path.read_text()
    assert HOOK_MARKER in content
    assert "git-why" in content
