# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# YouTube transcript fetch. No API key.

import logging
import re
import time
import urllib.parse
from typing import Any, Optional, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kgot.tools.base import ToolOutput
from kgot.tools.tools_v2_3.http_utils import (
    KGOT_USER_AGENT,
    build_retry_session,
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.YouTubeTranscriptTool")

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_ID_PATTERN = re.compile(
    r"(?:https?://)?(?:(?:www|m|music)\.youtube\.com/"
    r"(?:watch\?(?:[^&\s]*&)*v=|shorts/|live/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)

try:
    from youtube_transcript_api import (  # type: ignore
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    _TRANSCRIPT_ERRORS: tuple[type[BaseException], ...] = (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except ImportError:
    try:
        from youtube_transcript_api._errors import (  # type: ignore
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        _TRANSCRIPT_ERRORS = (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable)
    except ImportError:
        _TRANSCRIPT_ERRORS = ()


class YouTubeTranscriptSchema(BaseModel):
    video: str = Field(description="YouTube video URL or 11-character video ID.")
    languages: Optional[list[str]] = Field(
        default=None,
        description="Preferred caption languages in priority order (default ['en']).",
    )
    max_chars: int = Field(
        default=8000,
        description="Maximum transcript characters to return (1-100000).",
    )


def extract_video_id(video: str) -> str | None:
    """Extract an 11-character YouTube video ID from a URL or raw ID."""
    text = (video or "").strip()
    if not text:
        return None
    if _VIDEO_ID_PATTERN.match(text):
        return text
    match = _URL_ID_PATTERN.search(text)
    if match:
        return match.group(1)
    try:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(text).query)
        candidate = (params.get("v") or [""])[0]
        if _VIDEO_ID_PATTERN.match(candidate):
            return candidate
    except Exception:
        pass
    return None


def _fetch_transcript_items(video_id: str, languages: list[str]) -> list:
    """Fetch transcript items across old and new youtube-transcript-api shapes."""
    from youtube_transcript_api import YouTubeTranscriptApi

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    if hasattr(fetched, "to_raw_data"):
        return fetched.to_raw_data()
    return list(fetched)


def _snippet_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(getattr(item, "text", "") or "")


class YouTubeTranscriptTool(BaseTool):
    name: str = "youtube_transcript"
    description: str = (
        "Fetch the transcript/captions of a YouTube video as plain text. "
        "Use for video content, lectures, and interviews. No API key needed."
    )
    args_schema: Type[BaseModel] = YouTubeTranscriptSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    @collect_stats("youtube_transcript")
    def _run(
        self,
        video: str,
        languages: Optional[list[str]] = None,
        max_chars: int = 8000,
    ) -> str:
        start_time = time.time()

        video_id = extract_video_id(video)
        if not video_id:
            return ToolOutput.error_result(
                error=f"Could not parse a YouTube video ID from '{truncate(video, 100)}'.",
                error_code="VALIDATION_ERROR",
            ).to_llm_string()

        langs = languages or ["en"]
        limit = max(1, min(int(max_chars or 8000), 100000))
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        title = ""
        author = ""
        try:
            session = build_retry_session()
            oembed = session.get(
                "https://www.youtube.com/oembed",
                params={"url": video_url, "format": "json"},
                headers={"User-Agent": KGOT_USER_AGENT},
                timeout=10,
            )
            if oembed.status_code == 200:
                meta = oembed.json()
                title = meta.get("title", "") or ""
                author = meta.get("author_name", "") or ""
        except requests.exceptions.RequestException:
            pass  # Metadata is best-effort; the transcript still stands.

        try:
            items = _fetch_transcript_items(video_id, langs)
        except _TRANSCRIPT_ERRORS as e:
            return ToolOutput.error_result(
                error=(
                    f"No transcript available for '{video_url}': "
                    f"{e or 'captions disabled or unavailable'}."
                ),
                error_code="RESOURCE_NOT_FOUND",
                metadata={"video_url": video_url},
            ).to_llm_string()
        except Exception as e:
            message = str(e) or e.__class__.__name__
            lowered = message.lower()
            if any(
                token in lowered
                for token in (
                    "no transcript",
                    "transcriptsdisabled",
                    "notranscriptfound",
                    "captions disabled",
                    "no captions",
                    "video unavailable",
                )
            ):
                return ToolOutput.error_result(
                    error=f"No transcript available for '{video_url}'.",
                    error_code="RESOURCE_NOT_FOUND",
                    metadata={"video_url": video_url},
                ).to_llm_string()
            return ToolOutput.error_result(
                error=f"YouTube transcript fetch failed: {truncate(message, 300)}",
                error_code="API_ERROR",
                metadata={"video_url": video_url},
            ).to_llm_string()

        text = truncate(
            " ".join(t for t in (_snippet_text(i) for i in items) if t).strip(), limit
        )
        duration = time.time() - start_time
        if not text:
            return ToolOutput.success_result(
                result=f"No transcript text returned for '{video_url}'.",
                metadata={
                    "video_url": video_url,
                    "chars": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()

        header = f"**{title or video_id}**" + (f" by {author}" if author else "")
        return ToolOutput.success_result(
            result=f"{header}\nURL: {video_url}\n\n{text}",
            metadata={
                "video_url": video_url,
                "title": title,
                "author": author,
                "chars": len(text),
                "duration": round(duration, 2),
                "tool": "youtube_transcript",
            },
        ).to_llm_string()
