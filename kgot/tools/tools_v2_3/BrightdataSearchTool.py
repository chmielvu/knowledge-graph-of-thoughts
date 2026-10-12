# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Bright Data SERP search: Google results through a dedicated SERP zone.

import logging
import os
import time
import urllib.parse
from typing import Any, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kgot.tools.base import ToolOutput
from kgot.tools.tools_v2_3.http_utils import (
    KGOT_USER_AGENT,
    build_retry_session,
    redact_secrets,
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.BrightdataSearchTool")

_BRIGHTDATA_REQUEST_URL = "https://api.brightdata.com/request"
_SIGNUP_URL = "https://brightdata.com"


class BrightdataSearchSchema(BaseModel):
    query: str = Field(description="The web search query.")
    num_results: int = Field(
        default=10,
        description="Number of results to return (1-20).",
    )
    country: str = Field(
        default="us",
        description="Two-letter country code for geo-targeted results (e.g. 'us', 'uk').",
    )
    language: str = Field(
        default="en",
        description="Two-letter language code for results (e.g. 'en', 'de').",
    )


class BrightdataSearchTool(BaseTool):
    name: str = "brightdata_search"
    description: str = (
        "Search the web through Bright Data SERP (Google) and return ranked results "
        "with titles, URLs, and text snippets. Use for current information, news, and "
        "detailed search results. Needs BRIGHT_DATA_API_TOKEN and a dedicated SERP zone."
    )
    args_schema: Type[BaseModel] = BrightdataSearchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    def _missing_config_error(self, token: str | None, zone: str | None) -> str:
        missing = []
        if not token:
            missing.append("BRIGHT_DATA_API_TOKEN (or BRIGHT_DATA_API_KEY)")
        if not zone:
            missing.append("BRIGHTDATA_SERP_ZONE")
        return ToolOutput.error_result(
            error=(
                f"Bright Data SERP is not configured (missing: {', '.join(missing)}). "
                f"Set them in your environment; get a token at {_SIGNUP_URL}. "
                "Use a dedicated SERP zone, not the Web Unlocker zone."
            ),
            error_code="NOT_INITIALIZED",
        ).to_llm_string()

    @collect_stats("brightdata_search")
    def _run(
        self,
        query: str,
        num_results: int = 10,
        country: str = "us",
        language: str = "en",
    ) -> str:
        start_time = time.time()

        token = os.environ.get("BRIGHT_DATA_API_TOKEN") or os.environ.get(
            "BRIGHT_DATA_API_KEY"
        )
        zone = os.environ.get("BRIGHTDATA_SERP_ZONE", "serp")
        if not token or not zone:
            return self._missing_config_error(token, zone)

        count = max(1, min(int(num_results or 10), 20))
        google_url = (
            "https://www.google.com/search"
            f"?q={urllib.parse.quote_plus(query)}&num={count}"
            f"&gl={urllib.parse.quote(country or 'us')}"
            f"&hl={urllib.parse.quote(language or 'en')}&brd_json=1"
        )

        session = build_retry_session()
        try:
            response = session.post(
                _BRIGHTDATA_REQUEST_URL,
                json={"zone": zone, "url": google_url, "format": "raw"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": KGOT_USER_AGENT,
                },
                timeout=60,
            )
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="Bright Data SERP request timed out after 60 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100]},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"Bright Data SERP request failed: {redact_secrets(str(e))}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        if response.status_code == 429:
            return ToolOutput.error_result(
                error="Bright Data SERP rate limited (429). Retry later or lower request volume.",
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": 429},
            ).to_llm_string()
        if 400 <= response.status_code < 500:
            return ToolOutput.error_result(
                error=(
                    f"Bright Data SERP rejected the request (HTTP {response.status_code}). "
                    "Check the token and that BRIGHTDATA_SERP_ZONE is a dedicated SERP zone."
                ),
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": response.status_code},
            ).to_llm_string()
        if response.status_code != 200:
            return ToolOutput.error_result(
                error=f"Bright Data SERP failed (HTTP {response.status_code}).",
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": response.status_code},
            ).to_llm_string()

        try:
            payload = response.json()
        except ValueError:
            return ToolOutput.error_result(
                error="Bright Data SERP returned a non-JSON response.",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        organic = []
        if isinstance(payload.get("organic"), list):
            organic = payload["organic"]
        elif isinstance(payload.get("results"), list):
            organic = payload["results"]
        elif isinstance(payload.get("general"), dict) and isinstance(
            payload["general"].get("organic"), list
        ):
            organic = payload["general"]["organic"]

        items = []
        for entry in organic[:count]:
            if not isinstance(entry, dict):
                continue
            link = entry.get("link") or entry.get("url") or entry.get("href") or ""
            title = entry.get("title") or entry.get("name") or ""
            snippet = entry.get("description") or entry.get("snippet") or ""
            date = entry.get("date") or entry.get("posted") or ""
            if not link and not title:
                continue
            items.append(
                {"title": title, "url": link, "snippet": snippet, "date": date}
            )

        duration = time.time() - start_time
        if not items:
            return ToolOutput.success_result(
                result=f"No Bright Data search results for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "count": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()

        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item['title'] or '(no title)'}")
            lines.append(f"   URL: {item['url']}")
            if item["snippet"]:
                lines.append(f"   {truncate(item['snippet'], 400)}")
            if item["date"]:
                lines.append(f"   Date: {item['date']}")
        return ToolOutput.success_result(
            result="\n".join(lines),
            metadata={
                "query": query[:100],
                "count": len(items),
                "duration": round(duration, 2),
                "tool": "brightdata_search",
            },
        ).to_llm_string()
