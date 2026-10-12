# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# arXiv paper search. No API key. Respects the 3s request rule.

import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kgot.tools.base import ToolOutput
from kgot.tools.tools_v2_3.http_utils import (
    KGOT_USER_AGENT,
    build_retry_session,
    throttle,
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.ArxivSearchTool")

_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_SORT_VALUES = {"relevance", "lastUpdatedDate", "submittedDate"}
_ORDER_VALUES = {"ascending", "descending"}

_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivSearchSchema(BaseModel):
    query: str = Field(
        description="arXiv search query. Supports prefixes like 'au:Hinton' or 'cat:cs.LG'."
    )
    max_results: int = Field(
        default=10,
        description="Number of papers to return (1-50).",
    )
    sort_by: str = Field(
        default="relevance",
        description="Sort: 'relevance', 'lastUpdatedDate', or 'submittedDate'.",
    )
    sort_order: str = Field(
        default="descending",
        description="Sort order: 'ascending' or 'descending'.",
    )


def _fold_whitespace(text: str | None) -> str:
    return " ".join((text or "").split())


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Search arXiv preprints by keyword, author (au:), or category (cat:). "
        "Good for CS/math/physics papers. No API key needed."
    )
    args_schema: Type[BaseModel] = ArxivSearchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    @collect_stats("arxiv_search")
    def _run(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> str:
        start_time = time.time()

        sort_by = sort_by or "relevance"
        if sort_by not in _SORT_VALUES:
            return ToolOutput.error_result(
                error=f"Invalid sort_by '{sort_by}'. Use one of: relevance, lastUpdatedDate, submittedDate.",
                error_code="VALIDATION_ERROR",
            ).to_llm_string()
        sort_order = sort_order or "descending"
        if sort_order not in _ORDER_VALUES:
            return ToolOutput.error_result(
                error=f"Invalid sort_order '{sort_order}'. Use 'ascending' or 'descending'.",
                error_code="VALIDATION_ERROR",
            ).to_llm_string()

        count = max(1, min(int(max_results or 10), 50))
        throttle(_ARXIV_API_URL)
        session = build_retry_session()
        try:
            response = session.get(
                _ARXIV_API_URL,
                params={
                    "search_query": query,
                    "start": 0,
                    "max_results": count,
                    "sortBy": sort_by,
                    "sortOrder": sort_order,
                },
                headers={"User-Agent": KGOT_USER_AGENT},
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="arXiv search timed out after 30 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100]},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"arXiv search failed: {e}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            return ToolOutput.error_result(
                error=f"arXiv returned an unreadable response: {e}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        entries = root.findall("atom:entry", _NAMESPACES)
        duration = time.time() - start_time
        if not entries:
            return ToolOutput.success_result(
                result=f"No arXiv results for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "count": 0,
                    "duration": round(duration, 2),
                },
            ).to_llm_string()

        sections = []
        for entry in entries[:count]:
            entry_id = _fold_whitespace(
                (entry.findtext("atom:id", namespaces=_NAMESPACES) or "")
            )
            abs_url = (
                entry_id.replace("http://arxiv.org/", "https://arxiv.org/")
                if entry_id
                else ""
            )
            for link in entry.findall("atom:link", _NAMESPACES):
                if link.get("rel") == "alternate" and link.get("href"):
                    abs_url = link.get("href", "")
                    break
            title = _fold_whitespace(
                entry.findtext("atom:title", namespaces=_NAMESPACES)
            )
            summary = _fold_whitespace(
                entry.findtext("atom:summary", namespaces=_NAMESPACES)
            )
            published = _fold_whitespace(
                entry.findtext("atom:published", namespaces=_NAMESPACES)
            )
            primary = entry.find("arxiv:primary_category", _NAMESPACES)
            category = (primary.get("term") if primary is not None else "") or ""
            authors = [
                _fold_whitespace(a.findtext("atom:name", namespaces=_NAMESPACES))
                for a in entry.findall("atom:author", _NAMESPACES)
            ]
            authors = [a for a in authors if a]
            section = f"**{title or '(no title)'}**"
            if authors:
                section += f"\nAuthors: {', '.join(authors[:10])}"
            if published:
                section += f"\nPublished: {published[:10]}"
            if category:
                section += f"\nCategory: {category}"
            if abs_url:
                section += f"\nURL: {abs_url}"
            if summary:
                section += f"\n{truncate(summary, 600)}"
            sections.append(section)

        duration = time.time() - start_time
        return ToolOutput.success_result(
            result="\n\n---\n\n".join(sections),
            metadata={
                "query": query[:100],
                "count": len(sections),
                "duration": round(duration, 2),
                "tool": "arxiv_search",
            },
        ).to_llm_string()
