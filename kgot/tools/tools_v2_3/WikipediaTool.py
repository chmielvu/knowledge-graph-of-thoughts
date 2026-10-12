# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Wikipedia search: candidate discovery plus REST summaries. No API key.

import logging
import re
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
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.WikipediaTool")

_LANG_PATTERN = re.compile(r"^[a-z-]{2,12}$")


class WikipediaSearchSchema(BaseModel):
    query: str = Field(description="The topic or phrase to look up on Wikipedia.")
    lang: str = Field(
        default="en",
        description="Wikipedia language edition (e.g. 'en', 'de', 'fr').",
    )
    max_results: int = Field(
        default=3,
        description="Number of article summaries to return (1-10).",
    )


class WikipediaSearchTool(BaseTool):
    name: str = "wikipedia_search"
    description: str = (
        "Look up encyclopedic information on Wikipedia. Good for definitions and "
        "general background on a topic, person, place, or event. No API key needed."
    )
    args_schema: Type[BaseModel] = WikipediaSearchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    @collect_stats("wikipedia_search")
    def _run(self, query: str, lang: str = "en", max_results: int = 3) -> str:
        start_time = time.time()

        lang = (lang or "en").lower()
        if not _LANG_PATTERN.match(lang):
            return ToolOutput.error_result(
                error=f"Invalid Wikipedia language '{lang}'. Use a short code like 'en' or 'de'.",
                error_code="VALIDATION_ERROR",
            ).to_llm_string()

        count = max(1, min(int(max_results or 3), 10))
        session = build_retry_session()
        headers = {"User-Agent": KGOT_USER_AGENT}

        try:
            discovery = session.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": count,
                    "format": "json",
                    "origin": "*",
                },
                headers=headers,
                timeout=10,
            )
            discovery.raise_for_status()
            hits = discovery.json().get("query", {}).get("search", []) or []
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="Wikipedia discovery timed out after 10 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100]},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"Wikipedia discovery failed: {e}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        titles = [hit.get("title") for hit in hits if hit.get("title")][:count]
        duration = time.time() - start_time
        if not titles:
            return ToolOutput.success_result(
                result=f"No Wikipedia results for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "hits": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()

        sections = []
        for title in titles:
            try:
                summary = session.get(
                    "https://{}.wikipedia.org/api/rest_v1/page/summary/{}?redirect=true".format(
                        lang, urllib.parse.quote(title.replace(" ", "_"), safe="")
                    ),
                    headers=headers,
                    timeout=10,
                )
            except requests.exceptions.RequestException:
                continue
            if summary.status_code != 200:
                continue
            data = summary.json()
            if data.get("title") == "Not found.":
                continue
            extract = data.get("extract") or "No summary available."
            page_url = (
                data.get("content_urls", {}).get("desktop", {}).get("page")
                or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            )
            section = f"**{data.get('title', title)}**\n{truncate(extract, 600)}\nURL: {page_url}"
            if "may refer to" in extract:
                section += "\nNote: this looks like a disambiguation page; refine the query to a specific article."
            sections.append(section)

        duration = time.time() - start_time
        if not sections:
            return ToolOutput.success_result(
                result=f"No Wikipedia summaries available for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "hits": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()
        return ToolOutput.success_result(
            result="\n\n---\n\n".join(sections),
            metadata={
                "query": query[:100],
                "hits": len(sections),
                "duration": round(duration, 2),
                "tool": "wikipedia_search",
            },
        ).to_llm_string()
