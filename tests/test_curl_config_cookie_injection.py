"""Regression tests for curl --config injection via Netscape cookie values."""

import json
import re
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pytest

from agent.config import Config
from agent.curl_config import escape_curl_config_quoted_string
from agent.detection import _curl_netscape_cookie_config, _unlink_curl_netscape_sidecar
from agent.pipelines.instagram import _fetch_ig_comment
from agent.webhook import _netscape_cookie_upload_error


@pytest.fixture
def config(tmp_path):
    vault_path = tmp_path / "vault"
    temp_dir = tmp_path / "tmp"
    vault_path.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=vault_path,
        temp_dir=temp_dir,
        cookie_dir=tmp_path / "cookies",
        ig_carousel_min_size=100,
        minio_bucket="kb-agent-archive",
        minio_endpoint="http://localhost:9000",
        minio_tailscale_base="http://minio.test/base",
    )


def _netscape_line(value: str, name: str = "sessionid") -> str:
    return "\t".join(
        [".instagram.com", "TRUE", "/", "TRUE", "0", name, value]
    )


def _count_config_directive_lines(conf_text: str) -> int:
    return len([ln for ln in conf_text.splitlines() if ln.strip()])


def _parse_quoted_config_assignments(conf_text: str) -> list[tuple[str, str]]:
    """Naive parser for ``key = "value"`` lines (supports backslash escapes)."""
    out: list[tuple[str, str]] = []
    pattern = re.compile(r'^(\S+)\s*=\s*"(.*)"\s*$')
    for line in conf_text.splitlines():
        line = line.strip()
        if not line or line == "silent":
            continue
        if "=" not in line:
            continue
        m = pattern.match(line)
        if not m:
            out.append((line.split("=", 1)[0].strip(), ""))
            continue
        key, raw = m.group(1), m.group(2)
        # Unescape minimal set for assertions
        val = (
            raw.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )
        out.append((key, val))
    return out


class TestDetectionCurlNetscapeCookieConfig:
    def test_malicious_cookie_path_does_not_inject_directive(self, tmp_path):
        cookie_dir = tmp_path / 'cookies"evil'
        cookie_dir.mkdir()
        cookie = cookie_dir / 'threads".txt'
        cookie.write_text("session=1\n", encoding="utf-8")
        args, conf_path = _curl_netscape_cookie_config(cookie)
        try:
            assert conf_path is not None
            text = conf_path.read_text(encoding="utf-8")
            assert _count_config_directive_lines(text) == 1
            assert "https://evil.example/injected" not in text
            assert "\nurl = " not in text
            assert args == ["--config", str(conf_path)]
        finally:
            _unlink_curl_netscape_sidecar(conf_path)


class TestEscapeCurlConfigQuotedString:
    def test_escapes_quote_newline_and_backslash(self):
        raw = 'evil"\nurl = "http://bad"\tx'
        escaped = escape_curl_config_quoted_string(raw)
        assert "\nurl = " not in escaped
        assert escaped.count('"') == len([c for c in escaped if c == '"'])
        assert "\\n" in escaped


class TestFetchIgCommentCurlConfigInjection:
    @patch("agent.pipelines.instagram.subprocess.run")
    def test_malicious_sessionid_does_not_inject_directive(self, mock_run, config):
        """Cookie values with quotes/newlines must stay inside one header line."""
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        payload = 'sess"\nurl = "https://evil.example/injected"\n#"'
        (cookie_dir / "ig_cookies.txt").write_text(
            _netscape_line(payload, "sessionid") + "\n"
            + _netscape_line('csrf"\nmax-time = "99999"', "csrftoken") + "\n",
            encoding="utf-8",
        )
        captured: list[str] = []

        def _capture(cmd, **kwargs):
            conf = Path(cmd[cmd.index("--config") + 1])
            captured.append(conf.read_text(encoding="utf-8"))
            return MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "status": "ok",
                    "comments": [{
                        "pk": 111,
                        "text": "hi",
                        "user": {"username": "u"},
                    }],
                    "caption": {"text": "cap", "user": {"username": "p"}},
                }),
            )

        mock_run.side_effect = _capture
        _fetch_ig_comment(
            "https://www.instagram.com/reel/ShortCode1/?comment_id=111",
            config,
        )
        assert len(captured) == 1
        conf = captured[0]
        assert _count_config_directive_lines(conf) == 7
        assert "https://evil.example/injected" not in conf
        headers = [
            v for k, v in _parse_quoted_config_assignments(conf) if k == "header"
        ]
        assert any(h.startswith("Cookie: sessionid=") for h in headers)
        assert any(h.startswith("x-csrftoken: ") for h in headers)
        for h in headers:
            assert "\nurl = " not in h


class TestNetscapeCookieUploadRejectsUnsafeValues:
    def test_rejects_value_with_double_quote(self):
        body = (
            "# Netscape HTTP Cookie File\n"
            + _netscape_line('bad"token') + "\n"
        )
        err = _netscape_cookie_upload_error(body.encode())
        assert err is not None
        assert "disallowed" in err.lower() or '"' in err

    def test_rejects_value_with_newline(self):
        body = (
            "# Netscape HTTP Cookie File\n"
            + _netscape_line("line1\nurl = evil") + "\n"
        )
        err = _netscape_cookie_upload_error(body.encode())
        assert err is not None

    def test_rejects_malformed_tab_structure(self):
        body = "# Netscape HTTP Cookie File\n.only.com\tTRUE\t/\n"
        err = _netscape_cookie_upload_error(body.encode())
        assert err is not None

    def test_accepts_normal_netscape_line(self):
        body = (
            "# Netscape HTTP Cookie File\n"
            + _netscape_line("valid-session-token-123") + "\n"
        )
        assert _netscape_cookie_upload_error(body.encode()) is None
