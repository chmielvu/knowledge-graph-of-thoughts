# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Semantic Scholar paper search. Key optional; keyless works at lower limits.

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
    redact_secrets,
    truncate,
)
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.SemanticScholarSearchTool")

_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,venue,citationCount,abstract,externalIds,openAccessPdf"
_MAX_RETRY_AFTER_SECONDS = 30


class SemanticScholarSearchSchema(BaseModel):
    query: str = Field(description="The plain-text paper search query.")
    max_results: int = Field(
        default=10,
        description="Number of papers to return (1-100).",
    )


class SemanticScholarSearchTool(BaseTool):
    name: str = "semantic_scholar_search"
    description: str = (
        "Search Semantic Scholar papers with citation counts, venues, and open-access "
        "PDF links. Good for citation-ranked literature. No API key needed; "
        "SEMANTIC_SCHOLAR_API_KEY raises rate limits."
    )
    args_schema: Type[BaseModel] = SemanticScholarSearchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    @collect_stats("semantic_scholar_search")
    def _run(self, query: str, max_results: int = 10) -> str:
        start_time = time.time()

        count = max(1, min(int(max_results or 10), 100))
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        headers = {"User-Agent": KGOT_USER_AGENT}
        if api_key:
            headers["x-api-key"] = api_key

        session = build_retry_session()
        params = {"query": query, "limit": count, "fields": _FIELDS}
        try:
            response = session.get(
                _SEARCH_URL, params=params, headers=headers, timeout=30
            )
            if response.status_code == 429:
                wait = 0.0
                try:
                    wait = float(response.headers.get("Retry-After", "0"))
                except (TypeError, ValueError):
                    wait = 0.0
                wait = max(0.0, min(wait, _MAX_RETRY_AFTER_SECONDS))
                if wait:
                    time.sleep(wait)
                    response = session.get(
                        _SEARCH_URL, params=params, headers=headers, timeout=30
                    )
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="Semantic Scholar search timed out after 30 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100]},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"Semantic Scholar search failed: {redact_secrets(str(e))}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        if response.status_code == 429:
            return ToolOutput.error_result(
                error=(
                    "Semantic Scholar rate limited (429). Retry later or set "
                    "SEMANTIC_SCHOLAR_API_KEY for higher limits."
                ),
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": 429},
            ).to_llm_string()
        if response.status_code in (401, 403):
            return ToolOutput.error_result(
                error=(
                    f"Semantic Scholar rejected the request (HTTP {response.status_code}). "
                    "Check SEMANTIC_SCHOLAR_API_KEY or run without it."
                ),
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": response.status_code},
            ).to_llm_string()
        if response.status_code != 200:
            return ToolOutput.error_result(
                error=f"Semantic Scholar search failed (HTTP {response.status_code}).",
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": response.status_code},
            ).to_llm_string()

        papers = response.json().get("data", []) or []
        duration = time.time() - start_time
        if not papers:
            return ToolOutput.success_result(
                result=f"No Semantic Scholar results for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "count": 0,
                    "duration": round(duration, 2),
                    "keyless": not api_key,
                },
            ).to_llm_string()

        sections = []
        for paper in papers[:count]:
            title = paper.get("title") or "(no title)"
            authors = [a.get("name", "") for a in (paper.get("authors") or [])]
            authors = [a for a in authors if a]
            year = paper.get("year") or ""
            venue = paper.get("venue") or ""
            cites = paper.get("citationCount", 0)
            doi = (paper.get("externalIds") or {}).get("DOI") or ""
            pdf = (paper.get("openAccessPdf") or {}).get("url") or ""
            abstract = paper.get("abstract") or ""
            section = f"**{title}**"
            if authors:
                section += f"\nAuthors: {', '.join(authors[:10])}"
            if year:
                section += f"\nYear: {year}"
            if venue:
                section += f"\nVenue: {venue}"
            section += f"\nCited by: {cites}"
            if doi:
                section += f"\nDOI: {doi}"
            if pdf:
                section += f"\nPDF: {pdf}"
            if abstract:
                section += f"\n{truncate(abstract, 600)}"
            sections.append(section)

        duration = time.time() - start_time
        return ToolOutput.success_result(
            result="\n\n---\n\n".join(sections),
            metadata={
                "query": query[:100],
                "count": len(sections),
                "duration": round(duration, 2),
                "tool": "semantic_scholar_search",
                "keyless": not api_key,
            },
        ).to_llm_string()
