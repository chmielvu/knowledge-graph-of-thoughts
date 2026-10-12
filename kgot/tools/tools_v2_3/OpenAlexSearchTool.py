# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# OpenAlex works search. Key optional; keyless works at a lower budget.

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

logger = logging.getLogger("Controller.OpenAlexSearchTool")

_WORKS_URL = "https://api.openalex.org/works"
_MAX_POSITION = 100000
_MAX_ABSTRACT_WORDS = 5000


class OpenAlexSearchSchema(BaseModel):
    query: str = Field(description="The keyword query to search scholarly works for.")
    max_results: int = Field(
        default=10,
        description="Number of works to return (1-200).",
    )
    search_filter: str | None = Field(
        default=None,
        description="Optional OpenAlex filter, e.g. 'type:article' or 'from_publication_date:2023-01-01'.",
    )
    sort: str = Field(
        default="cited_by_count:desc",
        description="Sort, e.g. 'cited_by_count:desc' or 'publication_date:desc'.",
    )


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild prose from OpenAlex's {word: [positions]} inverted index."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if (
                not isinstance(position, int)
                or position < 0
                or position > _MAX_POSITION
            ):
                continue
            pairs.append((position, str(word)))
    pairs.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in pairs[:_MAX_ABSTRACT_WORDS])


class OpenAlexSearchTool(BaseTool):
    name: str = "openalex_search"
    description: str = (
        "Search OpenAlex scholarly works with citation counts and open-access info. "
        "Good for broad literature coverage across disciplines. No API key needed; "
        "OPENALEX_API_KEY raises the budget."
    )
    args_schema: Type[BaseModel] = OpenAlexSearchSchema

    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

    def _request(
        self, session: requests.Session, params: dict, headers: dict
    ) -> requests.Response:
        return session.get(_WORKS_URL, params=params, headers=headers, timeout=30)

    @collect_stats("openalex_search")
    def _run(
        self,
        query: str,
        max_results: int = 10,
        search_filter: str | None = None,
        sort: str = "cited_by_count:desc",
    ) -> str:
        start_time = time.time()

        count = max(1, min(int(max_results or 10), 200))
        params: dict[str, Any] = {
            "search": query,
            "per-page": count,
            "sort": sort or "cited_by_count:desc",
        }
        if search_filter:
            params["filter"] = search_filter
        contact_email = os.environ.get("CONTACT_EMAIL")
        if contact_email:
            params["mailto"] = contact_email

        api_key = os.environ.get("OPENALEX_API_KEY")
        headers = {"User-Agent": KGOT_USER_AGENT, "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        session = build_retry_session()
        try:
            response = self._request(session, params, headers)
            if response.status_code in (401, 403) and api_key:
                headers.pop("Authorization", None)
                response = self._request(session, params, headers)
                response.metadata_note = "keyless"  # type: ignore[attr-defined]
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="OpenAlex search timed out after 30 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100]},
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"OpenAlex search failed: {redact_secrets(str(e))}",
                error_code="API_ERROR",
                metadata={"query": query[:100]},
            ).to_llm_string()

        if response.status_code in (401, 403):
            return ToolOutput.error_result(
                error=(
                    f"OpenAlex rejected the request (HTTP {response.status_code}). "
                    "Check OPENALEX_API_KEY or run keyless."
                ),
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": response.status_code},
            ).to_llm_string()
        if response.status_code == 429:
            return ToolOutput.error_result(
                error="OpenAlex rate limited (429). Retry later or set OPENALEX_API_KEY.",
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": 429},
            ).to_llm_string()
        if response.status_code != 200:
            return ToolOutput.error_result(
                error=f"OpenAlex search failed (HTTP {response.status_code}).",
                error_code="API_ERROR",
                metadata={"query": query[:100], "status": response.status_code},
            ).to_llm_string()

        results = response.json().get("results", []) or []
        duration = time.time() - start_time
        keyless = not api_key or getattr(response, "metadata_note", "") == "keyless"
        if not results:
            return ToolOutput.success_result(
                result=f"No OpenAlex results for '{truncate(query, 100)}'.",
                metadata={
                    "query": query[:100],
                    "count": 0,
                    "duration": round(duration, 2),
                    "keyless": keyless,
                },
            ).to_llm_string()

        sections = []
        for work in results[:count]:
            title = work.get("title") or work.get("display_name") or "(no title)"
            authors = [
                (a.get("author") or {}).get("display_name", "")
                for a in (work.get("authorships") or [])
            ]
            authors = [a for a in authors if a]
            year = work.get("publication_year") or ""
            cites = work.get("cited_by_count", 0)
            doi = work.get("doi") or ""
            work_id = work.get("id") or ""
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            section = f"**{title}**"
            if authors:
                section += f"\nAuthors: {', '.join(authors[:10])}"
            if year:
                section += f"\nYear: {year}"
            section += f"\nCited by: {cites}"
            if doi:
                section += f"\nDOI: {doi}"
            if work_id:
                section += f"\nURL: {work_id}"
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
                "tool": "openalex_search",
                "keyless": keyless,
            },
        ).to_llm_string()
