# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Bright Data fetch: single-page content through a Web Unlocker zone.

import logging
import os
import time
from typing import Any, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kgot.tools.base import ToolOutput
from kgot.tools.tools_v2_3.http_utils import (
    KGOT_USER_AGENT,
    build_retry_session,
    is_public_url,
    redact_secrets,
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.BrightdataFetchTool")

_BRIGHTDATA_REQUEST_URL = "https://api.brightdata.com/request"
_SIGNUP_URL = "https://brightdata.com"


class BrightdataFetchSchema(BaseModel):
    url: str = Field(description="The public http(s) URL to fetch.")
    max_chars: int = Field(
        default=8000,
        description="Maximum characters of page content to return (1-50000).",
    )


class BrightdataFetchTool(BaseTool):
    name: str = "brightdata_fetch"
    description: str = (
        "Fetch a single public webpage through Bright Data Web Unlocker and return "
        "its markdown content. Use for bot-protected pages normal fetching cannot read. "
        "Needs BRIGHT_DATA_API_TOKEN and a Web Unlocker zone."
    )
    args_schema: Type[BaseModel] = BrightdataFetchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    @collect_stats("brightdata_fetch")
    def _run(self, url: str, max_chars: int = 8000) -> str:
        start_time = time.time()

        token = os.environ.get("BRIGHT_DATA_API_TOKEN") or os.environ.get(
            "BRIGHT_DATA_API_KEY"
        )
        zone = os.environ.get("BRIGHTDATA_UNLOCKER_ZONE", "unlocker")
        if not token or not zone:
            missing = []
            if not token:
                missing.append("BRIGHT_DATA_API_TOKEN (or BRIGHT_DATA_API_KEY)")
            if not zone:
                missing.append("BRIGHTDATA_UNLOCKER_ZONE")
            return ToolOutput.error_result(
                error=(
                    f"Bright Data fetch is not configured (missing: {', '.join(missing)}). "
                    f"Set them in your environment; get a token at {_SIGNUP_URL}."
                ),
                error_code="NOT_INITIALIZED",
            ).to_llm_string()

        if not is_public_url(url):
            return ToolOutput.error_result(
                error=f"Refusing to fetch non-public URL: {truncate(url, 200)}",
                error_code="VALIDATION_ERROR",
            ).to_llm_string()

        limit = max(1, min(int(max_chars or 8000), 50000))
        session = build_retry_session()
        try:
            response = session.post(
                _BRIGHTDATA_REQUEST_URL,
                json={
                    "zone": zone,
                    "url": url,
                    "format": "raw",
                    "data_format": "markdown",
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": KGOT_USER_AGENT,
                },
                timeout=90,
            )
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="Bright Data fetch timed out after 90 seconds.",
                error_code="TIMEOUT",
                metadata={"url": truncate(url, 200)},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"Bright Data fetch failed: {redact_secrets(str(e))}",
                error_code="API_ERROR",
                metadata={"url": truncate(url, 200)},
            ).to_llm_string()

        if response.status_code == 429:
            return ToolOutput.error_result(
                error="Bright Data fetch rate limited (429). Retry later.",
                error_code="API_ERROR",
                metadata={"url": truncate(url, 200), "status": 429},
            ).to_llm_string()
        if 400 <= response.status_code < 500:
            return ToolOutput.error_result(
                error=(
                    f"Bright Data fetch rejected the request (HTTP {response.status_code}). "
                    "Check the token and that BRIGHTDATA_UNLOCKER_ZONE is a Web Unlocker zone."
                ),
                error_code="API_ERROR",
                metadata={"url": truncate(url, 200), "status": response.status_code},
            ).to_llm_string()
        if response.status_code != 200:
            return ToolOutput.error_result(
                error=f"Bright Data fetch failed (HTTP {response.status_code}).",
                error_code="API_ERROR",
                metadata={"url": truncate(url, 200), "status": response.status_code},
            ).to_llm_string()

        content = truncate(response.text or "", limit)
        duration = time.time() - start_time
        if not content:
            return ToolOutput.success_result(
                result=f"No content returned for '{truncate(url, 200)}'.",
                metadata={
                    "url": truncate(url, 200),
                    "chars": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()
        return ToolOutput.success_result(
            result=f"URL: {url}\n\n{content}",
            metadata={
                "url": truncate(url, 200),
                "chars": len(content),
                "duration": round(duration, 2),
                "tool": "brightdata_fetch",
            },
        ).to_llm_string()
