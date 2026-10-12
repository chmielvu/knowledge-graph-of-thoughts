# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# YouTube metadata search via the Data API v3.

import logging
import os
import re
import time
from typing import Any, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kgot.tools.base import ToolOutput
from kgot.tools.tools_v2_3.http_utils import (
    build_retry_session,
    redact_secrets,
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.YouTubeSearchTool")

_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
_CONSOLE_URL = "https://console.cloud.google.com"
_SAFE_SEARCH_VALUES = {"moderate", "none", "strict"}
_DURATION_PATTERN = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class YouTubeSearchSchema(BaseModel):
    query: str = Field(description="The video search query.")
    max_results: int = Field(
        default=10,
        description="Number of videos to return (1-50).",
    )
    region_code: str | None = Field(
        default=None,
        description="Optional two-letter region code (e.g. 'US').",
    )
    safe_search: str = Field(
        default="moderate",
        description="SafeSearch level: 'moderate', 'none', or 'strict'.",
    )


def parse_iso8601_duration(value: str | None) -> int | None:
    """Parse an ISO 8601 duration (PT#H#M#S) into seconds."""
    if not value:
        return None
    match = _DURATION_PATTERN.match(value)
    if not match:
        return None
    parts = match.groupdict()
    return (
        int(parts["days"] or 0) * 86400
        + int(parts["hours"] or 0) * 3600
        + int(parts["minutes"] or 0) * 60
        + int(parts["seconds"] or 0)
    )


class YouTubeSearchTool(BaseTool):
    name: str = "youtube_search"
    description: str = (
        "Search YouTube videos and return metadata (title, channel, duration, views, "
        "description). Needs YOUTUBE_API_KEY. Transcript tools work without an API key."
    )
    args_schema: Type[BaseModel] = YouTubeSearchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    @collect_stats("youtube_search")
    def _run(
        self,
        query: str,
        max_results: int = 10,
        region_code: str | None = None,
        safe_search: str = "moderate",
    ) -> str:
        start_time = time.time()

        api_key = os.environ.get("YOUTUBE_API_KEY") or os.environ.get(
            "YOUTUBE_DATA_API_KEY"
        )
        if not api_key:
            return ToolOutput.error_result(
                error=(
                    "YouTube Data API is not configured (missing YOUTUBE_API_KEY). "
                    f"Create one at {_CONSOLE_URL}. "
                    "Transcript tools work without an API key."
                ),
                error_code="NOT_INITIALIZED",
            ).to_llm_string()

        safe_search = (safe_search or "moderate").lower()
        if safe_search not in _SAFE_SEARCH_VALUES:
            return ToolOutput.error_result(
                error=f"Invalid safe_search '{safe_search}'. Use one of: moderate, none, strict.",
                error_code="VALIDATION_ERROR",
            ).to_llm_string()

        region = None
        if region_code:
            region = region_code.strip().upper()
            if len(region) != 2 or not region.isalpha():
                return ToolOutput.error_result(
                    error=f"Invalid region_code '{region_code}'. Use a two-letter code like 'US'.",
                    error_code="VALIDATION_ERROR",
                ).to_llm_string()

        count = max(1, min(int(max_results or 10), 50))
        session = build_retry_session()
        search_params: dict[str, Any] = {
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": count,
            "safeSearch": safe_search,
            "key": api_key,
        }
        if region:
            search_params["regionCode"] = region

        try:
            search_resp = session.get(_SEARCH_URL, params=search_params, timeout=30)
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="YouTube search timed out after 30 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100]},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"YouTube search failed: {redact_secrets(str(e))}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        if search_resp.status_code in (403, 429):
            return ToolOutput.error_result(
                error=(
                    f"YouTube API quota exhausted (HTTP {search_resp.status_code}). "
                    "Wait for the daily quota reset or use transcript tools instead."
                ),
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": search_resp.status_code},
            ).to_llm_string()
        if 400 <= search_resp.status_code < 500:
            return ToolOutput.error_result(
                error=(
                    f"YouTube search rejected the request (HTTP {search_resp.status_code}). "
                    "Check YOUTUBE_API_KEY."
                ),
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": search_resp.status_code},
            ).to_llm_string()
        if search_resp.status_code != 200:
            return ToolOutput.error_result(
                error=f"YouTube search failed (HTTP {search_resp.status_code}).",
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": search_resp.status_code},
            ).to_llm_string()

        items = search_resp.json().get("items", []) or []
        video_ids = [
            item.get("id", {}).get("videoId")
            for item in items
            if isinstance(item.get("id"), dict) and item["id"].get("videoId")
        ]
        duration = time.time() - start_time
        if not video_ids:
            return ToolOutput.success_result(
                result=f"No YouTube results for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "count": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()

        details: dict[str, dict] = {}
        try:
            videos_resp = session.get(
                _VIDEOS_URL,
                params={
                    "part": "contentDetails,statistics,snippet",
                    "id": ",".join(video_ids),
                    "key": api_key,
                },
                timeout=30,
            )
            if videos_resp.status_code == 200:
                for entry in videos_resp.json().get("items", []) or []:
                    details[entry.get("id", "")] = entry
        except requests.exceptions.RequestException:
            pass  # Enrichment is best-effort; search snippets still stand.

        lines = []
        for item in items:
            video_id = item.get("id", {}).get("videoId", "")
            snippet = item.get("snippet", {}) or {}
            extra = details.get(video_id, {})
            content = extra.get("contentDetails", {}) or {}
            stats = extra.get("statistics", {}) or {}
            seconds = parse_iso8601_duration(content.get("duration"))
            lines.append(f"**{snippet.get('title') or '(no title)'}**")
            lines.append(f"   URL: https://www.youtube.com/watch?v={video_id}")
            lines.append(f"   Channel: {snippet.get('channelTitle') or 'unknown'}")
            if snippet.get("publishedAt"):
                lines.append(f"   Published: {snippet['publishedAt']}")
            if seconds is not None:
                lines.append(f"   Duration: {seconds}s")
            if stats.get("viewCount"):
                lines.append(f"   Views: {stats['viewCount']}")
            if snippet.get("description"):
                lines.append(f"   {truncate(snippet['description'], 300)}")

        duration = time.time() - start_time
        return ToolOutput.success_result(
            result="\n".join(lines),
            metadata={
                "query": query[:100],
                "count": len(video_ids),
                "duration": round(duration, 2),
                "tool": "youtube_search",
            },
        ).to_llm_string()
