"""Tests for agent.core.search — grep argument hardening."""

from unittest.mock import MagicMock, patch

from agent.core.search import search_notes


def test_search_uses_double_dash_before_pattern(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Knowledge base").mkdir(parents=True)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        search_notes("-injected-pattern", vault)
        argv = mock_run.call_args[0][0]
        assert argv[1] == "-rlF"
        assert "--" in argv
        # the pattern follows --, so a leading dash is treated literally, not as an option
        assert argv[argv.index("--") + 1] == "-injected-pattern"
