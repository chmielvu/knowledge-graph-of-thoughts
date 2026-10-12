"""Tests for shared HTTP helpers and core tool fixes."""

from __future__ import annotations

import socket

from kgot.tools.tools_v2_3 import http_utils
from kgot.tools.tools_v2_3.http_utils import (
    build_retry_session,
    is_public_url,
    redact_secrets,
    strip_html_to_text,
    throttle,
    truncate,
)
def test_retry_session_only_retries_transient_statuses():
    session = build_retry_session()
    adapter = session.get_adapter("https://")
    retry = getattr(adapter, "max_retries", None)
    assert retry is not None
    assert 429 in retry.status_forcelist
    assert 500 in retry.status_forcelist
    assert 400 not in retry.status_forcelist
    assert 403 not in retry.status_forcelist
    assert 404 not in retry.status_forcelist
    assert session.headers["User-Agent"] == http_utils.KGOT_USER_AGENT


def test_throttle_sleeps_for_host_delay(monkeypatch):
    calls = []
    monkeypatch.setattr(http_utils.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(http_utils.time, "sleep", lambda s: calls.append(s))
    http_utils._last_call.clear()
    http_utils._last_call["export.arxiv.org"] = 99.0
    throttle("http://export.arxiv.org/api/query")
    assert calls and calls[0] == 2.0


def test_redact_secrets_hides_keys_and_bearers():
    text = "key=AIza1234567890abcdef&x=1 Bearer abcdefghijklmnop"
    redacted = redact_secrets(text)
    assert "AIza1234567890abcdef" not in redacted
    assert "abcdefghijklmnop" not in redacted
    assert "[REDACTED]" in redacted


def test_truncate_marks_cut():
    assert truncate("abcdef", 3) == "abc...[truncated]"
    assert truncate("ab", 3) == "ab"
    assert truncate("", 3) == ""


def test_strip_html_to_text_collapses():
    assert strip_html_to_text("<p>Hello  <b>World</b></p>") == "Hello World"
    assert strip_html_to_text("") == ""


def test_is_public_url_rejects_private_and_local(monkeypatch):
    def fake_getaddrinfo(host, _port):
        mapping = {
            "example.com": [(2, 1, 6, "", ("93.184.216.34", 0))],
            "local.test": [(2, 1, 6, "", ("127.0.0.1", 0))],
            "link.test": [(2, 1, 6, "", ("169.254.169.254", 0))],
        }
        if host not in mapping:
            raise socket.gaierror("no such host")
        return mapping[host]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert is_public_url("https://example.com/page") is True
    assert is_public_url("http://169.254.169.254/") is False
    assert is_public_url("http://127.0.0.1/") is False
    assert is_public_url("http://local.test/") is False
    assert is_public_url("http://link.test/") is False
    assert is_public_url("file:///etc/passwd") is False
    assert is_public_url("https://user:pass@example.com/") is False
    assert is_public_url("https://missing.invalid/") is False
