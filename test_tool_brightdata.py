"""Tests for Bright Data search and fetch tools (network mocked)."""

from __future__ import annotations

import requests

from kgot.tools.tools_v2_3.BrightdataFetchTool import BrightdataFetchTool
from kgot.tools.tools_v2_3.BrightdataSearchTool import BrightdataSearchTool
from kgot.utils import UsageStatistics


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", json_error=False):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self._json_error = json_error
        self.request_body = None

    def json(self):
        if self._json_error:
            raise ValueError("bad json")
        return self._payload


class _FakeSession:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._exc is not None:
            raise self._exc
        if self._response is not None:
            self._response.request_body = json
        return self._response


def _stats(tmp_path):
    return UsageStatistics(file_name=str(tmp_path / "stats.jsonl"))


def _env(monkeypatch, **values):
    for key in (
        "BRIGHT_DATA_API_TOKEN",
        "BRIGHT_DATA_API_KEY",
        "BRIGHTDATA_SERP_ZONE",
        "BRIGHTDATA_UNLOCKER_ZONE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_search_shapes_results_with_fallback_keys(monkeypatch, tmp_path):
    _env(monkeypatch, BRIGHT_DATA_API_TOKEN="tok", BRIGHTDATA_SERP_ZONE="serp")
    payload = {
        "results": [
            {"name": "Title A", "url": "https://a.example", "snippet": "Snip A", "posted": "2026-01-01"},
            {"title": "Title B", "href": "https://b.example"},
            "junk",
            {},
        ]
    }
    fake = _FakeSession(_FakeResponse(payload))
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.BrightdataSearchTool.build_retry_session", lambda: fake
    )
    tool = BrightdataSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("kgot", num_results=10)
    assert out.startswith("SUCCESS:")
    assert "Title A" in out and "https://a.example" in out and "Snip A" in out
    assert "Title B" in out and "https://b.example" in out
    assert "junk" not in out
    assert "brd_json=1" in fake.posts[0]["json"]["url"]
    assert fake.posts[0]["json"]["zone"] == "serp"


def test_search_zero_hits_is_success(monkeypatch, tmp_path):
    _env(monkeypatch, BRIGHT_DATA_API_TOKEN="tok", BRIGHTDATA_SERP_ZONE="serp")
    fake = _FakeSession(_FakeResponse({"organic": []}))
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.BrightdataSearchTool.build_retry_session", lambda: fake
    )
    tool = BrightdataSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("nothing")
    assert out.startswith("SUCCESS:")
    assert "No Bright Data search results" in out


def test_search_missing_token_names_vars(monkeypatch, tmp_path):
    _env(monkeypatch, BRIGHTDATA_SERP_ZONE="serp")
    tool = BrightdataSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("kgot")
    assert out.startswith("ERROR (NOT_INITIALIZED):")
    assert "BRIGHT_DATA_API_TOKEN" in out
    assert "brightdata.com" in out


def test_search_missing_zone_names_zone(monkeypatch, tmp_path):
    _env(monkeypatch, BRIGHT_DATA_API_TOKEN="tok", BRIGHTDATA_SERP_ZONE="")
    tool = BrightdataSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("kgot")
    assert "BRIGHTDATA_SERP_ZONE" in out


def test_search_timeout_and_terminal_4xx(monkeypatch, tmp_path):
    _env(monkeypatch, BRIGHT_DATA_API_TOKEN="tok", BRIGHTDATA_SERP_ZONE="serp")
    fake_timeout = _FakeSession(exc=requests.exceptions.Timeout())
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.BrightdataSearchTool.build_retry_session", lambda: fake_timeout
    )
    tool = BrightdataSearchTool(usage_statistics=_stats(tmp_path))
    assert "ERROR (TIMEOUT):" in tool._run("kgot")

    fake_400 = _FakeSession(_FakeResponse({}, status_code=400))
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.BrightdataSearchTool.build_retry_session", lambda: fake_400
    )
    out = tool._run("kgot")
    assert "ERROR (API_ERROR):" in out
    assert "400" in out


def test_fetch_rejects_non_public_urls(monkeypatch, tmp_path):
    _env(monkeypatch, BRIGHT_DATA_API_TOKEN="tok", BRIGHTDATA_UNLOCKER_ZONE="unlocker")
    tool = BrightdataFetchTool(usage_statistics=_stats(tmp_path))
    assert "ERROR (VALIDATION_ERROR):" in tool._run("http://169.254.169.254/")
    assert "ERROR (VALIDATION_ERROR):" in tool._run("file:///etc/passwd")
    assert "ERROR (VALIDATION_ERROR):" in tool._run("https://user:pass@example.com/")


def test_fetch_truncates_and_envelopes_missing_key(monkeypatch, tmp_path):
    _env(monkeypatch, BRIGHT_DATA_API_TOKEN="tok", BRIGHTDATA_UNLOCKER_ZONE="unlocker")
    fake = _FakeSession(_FakeResponse(text="x" * 100))
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.BrightdataFetchTool.build_retry_session", lambda: fake
    )
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.BrightdataFetchTool.is_public_url", lambda url: True
    )
    tool = BrightdataFetchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("https://example.com", max_chars=10)
    assert out.startswith("SUCCESS:")
    assert "...[truncated]" in out
    assert fake.posts[0]["json"]["data_format"] == "markdown"

    _env(monkeypatch, BRIGHTDATA_UNLOCKER_ZONE="unlocker")
    out = tool._run("https://example.com")
    assert out.startswith("ERROR (NOT_INITIALIZED):")
    assert "BRIGHT_DATA_API_TOKEN" in out
