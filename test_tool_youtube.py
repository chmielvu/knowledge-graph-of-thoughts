"""Tests for YouTube search and transcript tools (network mocked)."""

from __future__ import annotations

import requests

from kgot.tools.tools_v2_3 import YouTubeTranscriptTool as transcript_module
from kgot.tools.tools_v2_3.YouTubeSearchTool import YouTubeSearchTool, parse_iso8601_duration
from kgot.tools.tools_v2_3.YouTubeTranscriptTool import YouTubeTranscriptTool, extract_video_id
from kgot.utils import UsageStatistics


class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses=None, exc=None):
        self._responses = list(responses or [])
        self._exc = exc
        self.gets = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.gets.append({"url": url, "params": params})
        if self._exc is not None:
            raise self._exc
        return self._responses.pop(0)


def _stats(tmp_path):
    return UsageStatistics(file_name=str(tmp_path / "stats.jsonl"))


def _env(monkeypatch, **values):
    for key in ("YOUTUBE_API_KEY", "YOUTUBE_DATA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_duration_parse_and_video_id_shapes():
    assert parse_iso8601_duration("PT1H2M3S") == 3723
    assert parse_iso8601_duration("PT5M") == 300
    assert parse_iso8601_duration("bad") is None
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("nope") is None


def test_search_shapes_and_enriches(monkeypatch, tmp_path):
    _env(monkeypatch, YOUTUBE_API_KEY="key123")
    search = _FakeResponse(
        {
            "items": [
                {
                    "id": {"videoId": "dQw4w9WgXcQ"},
                    "snippet": {
                        "title": "Video A",
                        "channelTitle": "Chan",
                        "publishedAt": "2026-01-01T00:00:00Z",
                        "description": "Desc A",
                    },
                }
            ]
        }
    )
    videos = _FakeResponse(
        {
            "items": [
                {
                    "id": "dQw4w9WgXcQ",
                    "contentDetails": {"duration": "PT4M20S"},
                    "statistics": {"viewCount": "42"},
                }
            ]
        }
    )
    fake = _FakeSession([search, videos])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.YouTubeSearchTool.build_retry_session", lambda: fake
    )
    tool = YouTubeSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("music")
    assert out.startswith("SUCCESS:")
    assert "Video A" in out and "Duration: 260s" in out and "Views: 42" in out
    assert "key123" not in out
    assert fake.gets[0]["params"]["safeSearch"] == "moderate"


def test_search_missing_key_quota_and_validation(monkeypatch, tmp_path):
    _env(monkeypatch)
    tool = YouTubeSearchTool(usage_statistics=_stats(tmp_path))
    out = tool._run("music")
    assert out.startswith("ERROR (NOT_INITIALIZED):")
    assert "without an API key" in out

    _env(monkeypatch, YOUTUBE_API_KEY="key123")
    assert "ERROR (VALIDATION_ERROR):" in tool._run("music", safe_search="wild")
    assert "ERROR (VALIDATION_ERROR):" in tool._run("music", region_code="USA")

    fake_403 = _FakeSession([_FakeResponse({}, status_code=403)])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.YouTubeSearchTool.build_retry_session", lambda: fake_403
    )
    out = tool._run("music")
    assert "quota exhausted" in out

    fake_timeout = _FakeSession(exc=requests.exceptions.Timeout())
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.YouTubeSearchTool.build_retry_session", lambda: fake_timeout
    )
    assert "ERROR (TIMEOUT):" in tool._run("music")


def test_transcript_new_api_path_and_invalid_id(monkeypatch, tmp_path):
    class _Fetched:
        def to_raw_data(self):
            return [{"text": "hello"}, {"text": "world"}]

    class _Api:
        def fetch(self, video_id, languages=None):
            assert video_id == "dQw4w9WgXcQ"
            return _Fetched()

    monkeypatch.setattr(transcript_module, "YouTubeTranscriptApi", None, raising=False)
    import kgot.tools.tools_v2_3.YouTubeTranscriptTool as mod

    monkeypatch.setattr(mod, "_fetch_transcript_items", lambda vid, langs: _Api().fetch(vid, langs).to_raw_data())
    fake = _FakeSession([_FakeResponse({"title": "T", "author_name": "A"})])
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.YouTubeTranscriptTool.build_retry_session", lambda: fake
    )
    tool = YouTubeTranscriptTool(usage_statistics=_stats(tmp_path))
    out = tool._run("https://youtu.be/dQw4w9WgXcQ")
    assert out.startswith("SUCCESS:")
    assert "hello world" in out
    assert "**T** by A" in out

    assert "ERROR (VALIDATION_ERROR):" in tool._run("nope")


def test_transcript_disabled_and_oembed_fail_soft(monkeypatch, tmp_path):
    def _raise_disabled(video_id, languages):
        raise transcript_module._TRANSCRIPT_ERRORS[0]("disabled") if transcript_module._TRANSCRIPT_ERRORS else Exception("No transcript")

    import kgot.tools.tools_v2_3.YouTubeTranscriptTool as mod

    monkeypatch.setattr(mod, "_fetch_transcript_items", _raise_disabled)
    fake = _FakeSession(exc=requests.exceptions.Timeout())
    monkeypatch.setattr(
        "kgot.tools.tools_v2_3.YouTubeTranscriptTool.build_retry_session", lambda: fake
    )
    tool = YouTubeTranscriptTool(usage_statistics=_stats(tmp_path))
    out = tool._run("dQw4w9WgXcQ")
    assert "ERROR (RESOURCE_NOT_FOUND):" in out
