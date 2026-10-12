# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Hacker News search via the Algolia API. No API key.

import logging
import time
from typing import Any, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kgot.tools.base import ToolOutput
from kgot.tools.tools_v2_3.http_utils import (
    KGOT_USER_AGENT,
    build_retry_session,
    strip_html_to_text,
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.HackerNewsTool")

_ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
_COMMENT_HITS = 3
_COMMENT_MAX_DEPTH = 3
_COMMENT_MAX_CHILDREN = 3
_COMMENT_MAX_CHARS = 600


class HackerNewsSearchSchema(BaseModel):
    query: str = Field(
        description="The keyword query to search Hacker News stories for."
    )
    sort: str = Field(
        default="relevance",
        description="Result ordering: 'relevance' or 'date'.",
    )
    max_results: int = Field(
        default=8,
        description="Number of stories to return (1-40). First page only.",
    )
    min_points: int = Field(
        default=0,
        description="Minimum story points to keep signal high (0 disables).",
    )
    include_comments: bool = Field(
        default=True,
        description="Fetch top comments for the first few stories.",
    )


class HackerNewsSearchTool(BaseTool):
    name: str = "hackernews_search"
    description: str = (
        "Search Hacker News stories and read top comments. Good for community "
        "discussion, launch reactions, and technical opinions. No API key needed."
    )
    args_schema: Type[BaseModel] = HackerNewsSearchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    def _walk_comments(self, node: dict, depth: int, lines: list[str]) -> None:
        if depth > _COMMENT_MAX_DEPTH:
            return
        text = strip_html_to_text(node.get("text") or "")
        author = node.get("author") or "unknown"
        if text:
            indent = "  " * depth
            lines.append(f"{indent}{author}: {truncate(text, _COMMENT_MAX_CHARS)}")
        for child in (node.get("children") or [])[:_COMMENT_MAX_CHILDREN]:
            if isinstance(child, dict):
                self._walk_comments(child, depth + 1, lines)

    def _fetch_thread(self, session: requests.Session, object_id: str) -> list[str]:
        try:
            response = session.get(
                f"{_ALGOLIA_BASE}/items/{object_id}",
                headers={"User-Agent": KGOT_USER_AGENT},
                timeout=10,
            )
            response.raise_for_status()
            tree = response.json()
        except requests.exceptions.RequestException:
            return []
        lines: list[str] = []
        for child in (tree.get("children") or [])[:_COMMENT_MAX_CHILDREN]:
            if isinstance(child, dict):
                self._walk_comments(child, 1, lines)
        return lines

    @collect_stats("hackernews_search")
    def _run(
        self,
        query: str,
        sort: str = "relevance",
        max_results: int = 8,
        min_points: int = 0,
        include_comments: bool = True,
    ) -> str:
        start_time = time.time()

        endpoint = "search_by_date" if (sort or "").lower() == "date" else "search"
        count = max(1, min(int(max_results or 8), 40))
        params: dict[str, Any] = {"query": query, "tags": "story", "hitsPerPage": count}
        if min_points and int(min_points) > 0:
            params["numericFilters"] = f"points>{int(min_points)}"

        session = build_retry_session()
        try:
            response = session.get(
                f"{_ALGOLIA_BASE}/{endpoint}",
                params=params,
                headers={"User-Agent": KGOT_USER_AGENT},
                timeout=10,
            )
            response.raise_for_status()
            hits = response.json().get("hits", []) or []
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="Hacker News search timed out after 10 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100]},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"Hacker News search failed: {e}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        duration = time.time() - start_time
        if not hits:
            return ToolOutput.success_result(
                result=f"No Hacker News results for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "hits": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()

        sections = []
        for hit in hits[:count]:
            title = hit.get("title") or "(no title)"
            url = (
                hit.get("url")
                or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            )
            points = hit.get("points", 0)
            author = hit.get("author") or "unknown"
            created = hit.get("created_at") or ""
            comments = hit.get("num_comments", 0)
            section = (
                f"**{title}**\nURL: {url}\n"
                f"Points: {points} | Comments: {comments} | By: {author} | At: {created}"
            )
            if include_comments and hit.get("objectID"):
                thread = self._fetch_thread(session, str(hit["objectID"]))
                if thread:
                    section += "\nTop comments:\n" + "\n".join(
                        thread[: _COMMENT_HITS * 4]
                    )
            sections.append(section)

        duration = time.time() - start_time
        return ToolOutput.success_result(
            result="\n\n---\n\n".join(sections),
            metadata={
                "query": query[:100],
                "hits": len(sections),
                "duration": round(duration, 2),
                "tool": "hackernews_search",
            },
        ).to_llm_string()
