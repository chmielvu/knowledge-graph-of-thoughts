# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import logging
import os
import time
from typing import Any, Type

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from kgot.tools.base import ToolOutput
from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.PollinationsSearchTool")


class PollinationsSearchSchema(BaseModel):
    query: str = Field(description="The web search question or query.")
    model: str = Field(
        default="perplexity-fast",
        description="Search model to use. Supported: perplexity-fast, perplexity-reasoning, gemini-search.",
    )
    detailed: bool = Field(
        default=False,
        description="Whether to request a more detailed answer.",
    )


class PollinationsSearchTool(BaseTool):
    name: str = "search_web_answer"
    description: str = (
        "Search the web using a hosted search-answer model and return a sourced answer. "
        "Use this for synthesis-heavy web questions. For raw page navigation and step-by-step browsing, "
        "use the search agent or browser tools instead."
    )
    args_schema: Type[BaseModel] = PollinationsSearchSchema

    api_base_url: str = "https://gen.pollinations.ai"
    usage_statistics: UsageStatistics = None

    def __init__(self, usage_statistics: UsageStatistics, **kwargs: Any):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics
        self.api_base_url = os.environ.get("POLLINATIONS_API_BASE_URL", self.api_base_url).rstrip("/")

    @collect_stats("search_web_answer")
    def _run(self, query: str, model: str = "perplexity-fast", detailed: bool = False) -> str:
        start_time = time.time()

        api_key = os.environ.get("POLLINATIONS_API_KEY")
        if not api_key:
            return ToolOutput.error_result(
                error="Pollinations API key is missing. Configure POLLINATIONS_API_KEY to use this tool.",
                error_code="NOT_INITIALIZED"
            ).to_llm_string()

        if model not in {"perplexity-fast", "perplexity-reasoning", "gemini-search"}:
            return ToolOutput.error_result(
                error=f"Unsupported Pollinations search model: '{model}'. Use one of: perplexity-fast, perplexity-reasoning, gemini-search.",
                error_code="VALIDATION_ERROR"
            ).to_llm_string()

        system_prompt = (
            "Search the web and provide a comprehensive answer with sources. Include relevant details and cite your sources."
            if detailed
            else "Search the web and provide a concise, accurate answer. Include source URLs."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        time_before = time.time()
        try:
            response = requests.post(
                f"{self.api_base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return ToolOutput.error_result(
                error="Request timed out after 30 seconds.",
                error_code="TIMEOUT",
                metadata={"query": query[:100], "model": model}
            ).to_llm_string()
        except requests.exceptions.RequestException as e:
            return ToolOutput.error_result(
                error=f"API request failed: {str(e)}",
                error_code="API_ERROR",
                metadata={"query": query[:100], "model": model}
            ).to_llm_string()

        time_after = time.time()
        result = response.json()

        if self.usage_statistics is not None:
            usage = result.get("usage", {})
            self.usage_statistics.log_statistic(
                "PollinationsSearchTool._run",
                time_before,
                time_after,
                result.get("model", model),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                0,
            )

        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = result.get("citations", []) or []

        lines = [
            "### 1. Search outcome (short version):",
            answer,
            "",
            "### 2. Search outcome (extremely detailed version):",
            answer,
            "",
            "### 3. Additional context:",
            f"Model used: {result.get('model', model)}",
        ]
        if citations:
            lines.append("Sources:")
            for citation in citations:
                if isinstance(citation, dict):
                    url = citation.get("url") or citation.get("link") or str(citation)
                else:
                    url = str(citation)
                lines.append(f"- {url}")

        duration = time_after - start_time
        return ToolOutput.success_result(
            result="\n".join(lines),
            metadata={
                "model": result.get("model", model),
                "duration": round(duration, 2),
                "has_citations": bool(citations)
            }
        ).to_llm_string()
