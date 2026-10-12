"""Tests for Wikipedia and Hacker News tools (network mocked)."""

from __future__ import annotations

import requests

from kgot.tools.tools_v2_3.HackerNewsTool import HackerNewsSearchTool
from kgot.tools.tools_v2_3.WikipediaTool import WikipediaSearchTool
from kgot.utils import UsageStatistics


class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


class _FakeSession:
    def __init__(self, responses=None, exc=None):
        self._responses = list(responses or [])
        self._exc = exc
        self.gets = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append({"url": url, "params": params, "timeout": timeout})
        if self._exc is not None:
            raise self._exc
        return self._responses.pop(0)


def _stats(tmp_path):
    return UsageStatistics(file_name=str(tmp_path / "stats.jsonl"))


def test_wikipedia_shapes_discovery_plus_summary(monkeypatch, tmp_path):
    discovery = _FakeResponse({"query": {"search": [{"title": "KGoT"}]}})
    summary = _FakeResponse(
        {
            "title": "KGoT",
            "extract": "Knowledge Graph of Thoughts is great. " * 100,
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/KGoT"}},
        }
    )
    fake = _FakeSession([discovery, summary])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.WikipediaTool.build_retry_session", lambda: fake
    )
    tool = WikipediaSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("kgot")
    assert out.startswith("SUCCESS:")
    assert "**KGoT**" in out
    assert "https://en.wikipedia.org/wiki/KGoT" in out
    assert "...[truncated]" in out
    assert fake.gets[0]["params"]["list"] == "search"


def test_wikipedia_miss_and_disambiguation(monkeypatch, tmp_path):
    fake_empty = _FakeSession([_FakeResponse({"query": {"search": []}})])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.WikipediaTool.build_retry_session", lambda: fake_empty
    )
    tool = WikipediaSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("nothing-here")
    assert out.startswith("SUCCESS:")
    assert "No Wikipedia results" in out

    discovery = _FakeResponse({"query": {"search": [{"title": "Mercury"}]}})
    summary = _FakeResponse(
        {
            "title": "Mercury",
            "extract": "Mercury may refer to: a planet, an element.",
            "content_urls": {},
        }
    )
    fake = _FakeSession([discovery, summary])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.WikipediaTool.build_retry_session", lambda: fake
    )
    out = tool._run("mercury")
    assert "disambiguation page" in out
    assert "https://en.wikipedia.org/wiki/Mercury" in out


def test_wikipedia_validates_lang_and_timeout(monkeypatch, tmp_path):
    tool = WikipediaSearchTool(usage_statistics=_stats(tmp_path))
    assert "ERROR (VALIDATION_ERROR):" in tool._run("kgot", lang="xx!")
    fake = _FakeSession(exc=requests.exceptions.Timeout())
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.WikipediaTool.build_retry_session", lambda: fake
    )
    assert "ERROR (TIMEOUT):" in tool._run("kgot")


def test_hackernews_selects_endpoint_and_filters(monkeypatch, tmp_path):
    search = _FakeResponse(
        {
            "hits": [
                {
                    "title": "Story A",
                    "url": "https://a.example",
                    "points": 150,
                    "author": "alice",
                    "created_at": "2026-01-01T00:00:00Z",
                    "num_comments": 3,
                    "objectID": "1",
                }
            ]
        }
    )
    thread = _FakeResponse(
        {
            "children": [
                {
                    "author": "bob",
                    "text": "<p>Great <b>post</b></p>",
                    "children": [{"author": "carol", "text": "Agree", "children": []}],
                }
            ]
        }
    )
    fake = _FakeSession([search, thread])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.HackerNewsTool.build_retry_session", lambda: fake
    )
    tool = HackerNewsSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("kgot", sort="date", min_points=100, include_comments=True)
    assert out.startswith("SUCCESS:")
    assert "Story A" in out and "bob: Great post" in out
    assert "search_by_date" in fake.gets[0]["url"]
    assert fake.gets[0]["params"]["numericFilters"] == "points>100"
    assert "/items/1" in fake.gets[1]["url"]


def test_hackernews_zero_hits_and_timeout(monkeypatch, tmp_path):
    fake_empty = _FakeSession([_FakeResponse({"hits": []})])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.HackerNewsTool.build_retry_session", lambda: fake_empty
    )
    tool = HackerNewsSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("nothing")
    assert out.startswith("SUCCESS:")
    assert "No Hacker News results" in out

    fake_timeout = _FakeSession(exc=requests.exceptions.Timeout())
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.HackerNewsTool.build_retry_session", lambda: fake_timeout
    )
    assert "ERROR (TIMEOUT):" in tool._run("kgot")
