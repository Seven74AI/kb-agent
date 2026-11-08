"""Regression: detection/router/redact import graph must not circular-import."""

import subprocess
import sys


def test_detection_router_redact_import_in_fresh_subprocess():
    code = (
        "import agent.detection; import agent.router; "
        "import agent.core.redact; import agent.redact"
    )
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
