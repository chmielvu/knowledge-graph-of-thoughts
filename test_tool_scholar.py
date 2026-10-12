"""Tests for arXiv, OpenAlex, and Semantic Scholar tools (network mocked)."""

from __future__ import annotations

import requests

from kgot.tools.tools_v2_3 import ArxivSearchTool as arxiv_module
from kgot.tools.tools_v2_3.ArxivSearchTool import ArxivSearchTool
from kgot.tools.tools_v2_3.OpenAlexSearchTool import OpenAlexSearchTool, reconstruct_abstract
from kgot.tools.tools_v2_3.SemanticScholarSearchTool import SemanticScholarSearchTool
from kgot.utils import UsageStatistics


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

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
        self.gets.append({"url": url, "params": params, "headers": dict(headers or {})})
        if self._exc is not None:
            raise self._exc
        return self._responses.pop(0)


def _stats(tmp_path):
    return UsageStatistics(file_name=str(tmp_path / "stats.jsonl"))


def _env(monkeypatch, **values):
    for key in ("OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "CONTACT_EMAIL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>  A Test  Paper </title>
    <summary>  First line.
      Second line. </summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
  <entry>
    <title>Lonely</title>
  </entry>
</feed>
"""


def test_arxiv_parses_and_throttles(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(arxiv_module, "throttle", lambda url: calls.append(url))
    fake = _FakeSession([_FakeResponse(text=_ARXIV_FEED)])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.ArxivSearchTool.build_retry_session", lambda: fake
    )
    tool = ArxivSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("au:Hinton", sort_by="relevance")
    assert out.startswith("SUCCESS:")
    assert calls == ["http://export.arxiv.org/api/query"]
    assert "**A Test Paper**" in out
    assert "First line. Second line." in out
    assert "https://arxiv.org/abs/2301.00001v1" in out
    assert "**Lonely**" in out
    assert "ERROR (VALIDATION_ERROR):" in tool._run("x", sort_by="nope")


def test_arxiv_zero_hits_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(arxiv_module, "throttle", lambda url: None)
    empty = _FakeSession([_FakeResponse(text='<feed xmlns="http://www.w3.org/2005/Atom"/>')])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.ArxivSearchTool.build_retry_session", lambda: empty
    )
    tool = ArxivSearchTool(usage_statistics=_stats(tmp_path))
    assert "No arXiv results" in tool._run("nothing")

    fake_timeout = _FakeSession(exc=requests.exceptions.Timeout())
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.ArxivSearchTool.build_retry_session", lambda: fake_timeout
    )
    assert "ERROR (TIMEOUT):" in tool._run("x")


def test_reconstruct_abstract_guards_positions():
    assert reconstruct_abstract({"deep": [0], "learning": [1]}) == "deep learning"
    assert reconstruct_abstract({"bad": [10**9], "ok": [0]}) == "ok"
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_openalex_keyless_retry_and_shaping(monkeypatch, tmp_path):
    _env(monkeypatch, OPENALEX_API_KEY="secret", CONTACT_EMAIL="me@example.com")
    first = _FakeResponse({}, status_code=401)
    second = _FakeResponse(
        {
            "results": [
                {
                    "title": "Work A",
                    "authorships": [{"author": {"display_name": "Ann"}}],
                    "publication_year": 2024,
                    "cited_by_count": 7,
                    "doi": "https://doi.org/10.1/a",
                    "id": "https://openalex.org/W1",
                    "abstract_inverted_index": {"hello": [0], "world": [1]},
                }
            ]
        }
    )
    fake = _FakeSession([first, second])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.OpenAlexSearchTool.build_retry_session", lambda: fake
    )
    tool = OpenAlexSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("graphs", search_filter="type:article")
    assert out.startswith("SUCCESS:")
    assert "**Work A**" in out and "hello world" in out
    assert "Authorization" in fake.gets[0]["headers"]
    assert "Authorization" not in fake.gets[1]["headers"]
    assert fake.gets[0]["params"]["mailto"] == "me@example.com"
    assert fake.gets[0]["params"]["filter"] == "type:article"


def test_semantic_scholar_retry_after_capped(monkeypatch, tmp_path):
    _env(monkeypatch)
    waits = []
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.SemanticScholarSearchTool.time.sleep",
        lambda s: waits.append(s),
    )
    first = _FakeResponse({}, status_code=429, headers={"Retry-After": "3600"})
    second = _FakeResponse(
        {
            "data": [
                {
                    "title": "Paper A",
                    "authors": [{"name": "Ann"}],
                    "year": 2024,
                    "venue": "V",
                    "citationCount": 3,
                    "externalIds": {"DOI": "10.1/a"},
                    "openAccessPdf": {"url": "https://pdf.example/a"},
                    "abstract": "Abstract A",
                }
            ]
        }
    )
    fake = _FakeSession([first, second])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.SemanticScholarSearchTool.build_retry_session", lambda: fake
    )
    tool = SemanticScholarSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("graphs")
    assert out.startswith("SUCCESS:")
    assert waits == [30]
    assert "**Paper A**" in out and "PDF: https://pdf.example/a" in out

    fake_empty = _FakeSession([_FakeResponse({"data": []})])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.SemanticScholarSearchTool.build_retry_session", lambda: fake_empty
    )
    assert "No Semantic Scholar results" in tool._run("nothing")
