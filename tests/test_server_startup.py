"""Check startup ordering and fail-fast behavior without installing packages."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _bash():
    executable = shutil.which("bash")
    if executable:
        return executable
    git = shutil.which("git")
    if git and os.name == "nt":
        executable = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if executable.is_file():
            return str(executable)
    pytest.skip("Bash is required to test the server startup script")


@pytest.mark.parametrize("fail_at", [0, 1, 2, 3, 4, 5])
def test_server_startup(tmp_path, fail_at):
    script = (Path(__file__).resolve().parents[1] / "run_server.sh").read_text()
    (tmp_path / "continuous").mkdir()
    (tmp_path / "caching.log").touch()
    harness = """
calls=0
python() {
    calls=$((calls + 1))
    printf '%s\\n' "$*"
    if [ "$calls" -eq "$FAIL_AT" ]; then
        return 42
    fi
}
"""
    result = subprocess.run(
        [_bash(), "--noprofile", "--norc"],
        input=harness + script,
        cwd=tmp_path,
        env={**os.environ, "FAIL_AT": str(fail_at)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    calls = [
        line for line in result.stdout.splitlines()
        if line.startswith(("-m pip ", "-c ", "repoui.py "))
    ]
    assert "vtools3.git#egg=vtools" in calls[0]
    assert "--upgrade --no-deps" in calls[0]
    if fail_at:
        assert result.returncode == 42, result.stderr
        assert len(calls) == fail_at
        assert not any(call.startswith("repoui.py ") for call in calls)
    else:
        assert result.returncode == 0, result.stderr
        assert len(calls) == 6
        assert "dvue.git" in calls[1]
        assert "dms_datastore.git" in calls[2]
        assert "max_change_test" in calls[4]
        assert "from dms_datastore import auto_screen" in calls[4]
        assert calls[5].startswith("repoui.py ")
