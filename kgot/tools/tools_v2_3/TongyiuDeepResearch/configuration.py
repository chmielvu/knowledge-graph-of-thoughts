"""
Deep Research Configuration

Configuration for the research agent including:
- Search API settings with fallback chain (SearXNG -> DuckDuckGo)
- LLM settings (Tongyi orchestrator, Mistral extraction)
- Research workflow parameters
"""

from __future__ import annotations

import os
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class SearchAPI(str, Enum):
    """Available search API options."""
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    POLLINATIONS = "pollinations"


class DeepResearchConfig(BaseModel):
    """
    Configuration for the deep research agent.

    Supports fallback chain: SearXNG (public instances) -> HF Space -> DuckDuckGo
    """

    # ==========================================================================
    # LLM Settings
    # ==========================================================================

    tongyi_model: str = Field(
        default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        description="Tongyi model identifier for orchestration"
    )
    nanogpt_api_key: str = Field(
        default="",
        description="nanoGPT API key for Tongyi"
    )
    nanogpt_base_url: str = Field(
        default="https://nano-gpt.com/api/subscription/v1",
        description="nanoGPT API base URL"
    )

    mistral_api_key: str = Field(
        default="",
        description="Mistral API key for extraction"
    )
    mistral_model: str = Field(
        default="mistral-medium-2505",
        description="Mistral model for structured extraction"
    )

    # ==========================================================================
    # Search API Settings (with fallback chain)
    # ==========================================================================

    primary_search_api: SearchAPI = Field(
        default=SearchAPI.SEARXNG,
        description="Primary search API to use"
    )

    # Public SearXNG instances (tried in order)
    searxng_urls: List[str] = Field(
        default=[
            "https://searx.be",
            "https://search.bus-hit.me",
            "https://saneowl-searxng-search-engine.hf.space",
        ],
        description="SearXNG instances to try in order (fallback chain)"
    )

    fallback_to_duckduckgo: bool = Field(
        default=True,
        description="Fall back to DuckDuckGo if SearXNG fails"
    )

    include_wikipedia: bool = Field(
        default=True,
        description="Include Wikipedia search alongside web search"
    )

    max_search_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results per query"
    )

    # ==========================================================================
    # Research Workflow Settings
    # ==========================================================================

    max_research_loops: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum research iterations"
    )
    max_sources_per_query: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum sources to gather per search"
    )
    fetch_full_page: bool = Field(
        default=True,
        description="Fetch full page content for promising results"
    )

    # ==========================================================================
    # Crawling Settings
    # ==========================================================================

    default_crawl_depth: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Default depth for deep crawling"
    )
    default_max_pages: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Default max pages for deep crawling"
    )

    # ==========================================================================
    # Pollinations API (optional)
    # ==========================================================================

    pollinations_api_key: str = Field(
        default="",
        description="Pollinations API key (optional for free tier)"
    )
    pollinations_api_base: str = Field(
        default="https://text.pollinations.ai",
        description="Pollinations API base URL"
    )

    # ==========================================================================
    # Rate Limiting & Retry
    # ==========================================================================

    search_timeout: int = Field(
        default=10,
        description="Timeout for search API calls in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed API calls"
    )
    retry_backoff_base: float = Field(
        default=2.0,
        description="Base for exponential backoff"
    )

    @classmethod
    def from_env(cls) -> "DeepResearchConfig":
        """Create config from environment variables."""
        return cls(
            nanogpt_api_key=os.getenv("NANOGPT_API_KEY", ""),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
            pollinations_api_key=os.getenv("POLLINATIONS_API_KEY", ""),
        )

    def get_search_instances(self) -> List[str]:
        """
        Get ordered list of search instances to try.

        Returns SearXNG URLs followed by DDG fallback if enabled.
        """
        instances = []
        if self.primary_search_api == SearchAPI.SEARXNG:
            instances.extend(self.searxng_urls)
        if self.fallback_to_duckduckgo:
            instances.append("duckduckgo")
        return instances