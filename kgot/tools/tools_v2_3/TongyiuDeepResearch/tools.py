"""
Deep Research Tools

Tools for the research agent with:
- ResilientSearchTool: SearXNG -> DuckDuckGo fallback chain
- WikipediaTool: Wikipedia search via LangChain
- Best practice tool descriptions following tarun7r/deep-research-agent patterns
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Type

import httpx
from duckduckgo_search import DDGS
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .configuration import DeepResearchConfig

logger = logging.getLogger(__name__)


# =============================================================================
# User Agent Rotation (anti-bot bypass)
# =============================================================================

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]


def get_random_user_agent() -> str:
    """Get a random user agent for anti-bot bypass."""
    return random.choice(USER_AGENTS)


# =============================================================================
# Tool Input Schemas
# =============================================================================

class WebSearchInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(
        description="A well-crafted search query string. Be specific and include relevant context."
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )
    include_wikipedia: bool = Field(
        default=True,
        description="Also search Wikipedia for encyclopedic info"
    )


class WikipediaSearchInput(BaseModel):
    """Input for Wikipedia search tool."""
    query: str = Field(
        description="The topic or concept to search for on Wikipedia"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of Wikipedia pages to return"
    )


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    snippet: str
    source_type: str = "web"  # "web", "wikipedia", "ai_search"
    credibility_score: Optional[float] = None


# =============================================================================
# Resilient Search Tool (SearXNG -> DDG fallback)
# =============================================================================

class ResilientSearchTool(BaseTool):
    """
    Web search tool with fallback chain: SearXNG -> DuckDuckGo.

    Tries multiple SearXNG instances in order, falling back to DuckDuckGo
    if all fail. This provides resilience against rate limits and outages.
    """

    name: str = "web_search"
    description: str = """Search the web for authoritative information.

    Use this tool when:
    - Gathering factual information on any topic
    - Finding authoritative sources (academic, government, official docs)
    - Getting current information on recent events or trends

    Args:
        query: A well-crafted search query string. Be specific and include
               relevant context. Supports search operators like site:edu
        max_results: Maximum number of results to return (1-20, default 5)
        include_wikipedia: Also search Wikipedia for encyclopedic info (default True)

    Returns:
        List of search results with title, url, snippet fields

    Tips:
        - Use "site:edu" for academic sources
        - Add year for current info: "AI trends 2026"
        - Check URL domains: .edu, .gov, .org often indicate credibility
        - Combine with visit_page for thorough research

    Examples:
        GOOD: "site:arxiv.org transformer attention mechanism 2024"
        GOOD: "Neo4j vs TigerGraph performance benchmark comparison"
        GOOD: "Python asyncio best practices tutorial"
        AVOID: Single words like "AI" or overly broad queries
    """
    args_schema: Type[BaseModel] = WebSearchInput

    config: Optional[DeepResearchConfig] = None

    def __init__(self, config: Optional[DeepResearchConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or DeepResearchConfig.from_env()

    def _run(
        self,
        query: str,
        max_results: int = 5,
        include_wikipedia: bool = True
    ) -> str:
        """Synchronous search (wraps async)."""
        return asyncio.run(self._arun(query, max_results, include_wikipedia))

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        include_wikipedia: bool = True
    ) -> str:
        """
        Execute search with fallback chain.

        Tries SearXNG instances in order, falls back to DuckDuckGo.
        Optionally includes Wikipedia results.
        """
        results = []
        config = self.config or DeepResearchConfig.from_env()

        # Try SearXNG instances
        for instance_url in config.searxng_urls:
            try:
                searx_results = await self._searxng_search(
                    query, instance_url, max_results, config
                )
                if searx_results:
                    results.extend(searx_results)
                    logger.info(f"SearXNG search succeeded: {instance_url}")
                    break
            except Exception as e:
                logger.warning(f"SearXNG {instance_url} failed: {e}")
                continue

        # Fallback to DuckDuckGo
        if not results and config.fallback_to_duckduckgo:
            logger.info("Falling back to DuckDuckGo")
            ddg_results = await self._duckduckgo_search(query, max_results)
            results.extend(ddg_results)

        # Add Wikipedia results
        if include_wikipedia:
            wiki_results = await self._wikipedia_search(query)
            results.extend(wiki_results)

        if not results:
            return "No results found. Try:\n1) Different search terms\n2) Wait 60s and retry\n3) Use ai_web_search"

        return self._format_results(results)

    async def _searxng_search(
        self,
        query: str,
        instance_url: str,
        max_results: int,
        config: DeepResearchConfig
    ) -> List[SearchResult]:
        """
        Search using SearXNG API.

        Uses JSON API format with retry logic for rate limits.
        """
        search_endpoint = f"{instance_url}/search"
        params = {
            'q': query,
            'format': 'json',
            'results': str(max_results),
        }

        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'application/json',
        }

        for attempt in range(config.max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        search_endpoint,
                        params=params,
                        headers=headers,
                        timeout=config.search_timeout
                    )

                    if response.status_code == 429:
                        # Rate limited - exponential backoff
                        wait_time = (config.retry_backoff_base ** attempt +
                                    random.uniform(0, 1))
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()

                    if 'results' not in data or not data['results']:
                        return []

                    results = []
                    for item in data['results'][:max_results]:
                        results.append(SearchResult(
                            title=item.get('title', 'No Title'),
                            url=item.get('url', ''),
                            snippet=item.get('content', ''),
                            source_type="searxng"
                        ))
                    return results

            except httpx.HTTPStatusError as e:
                logger.warning(f"SearXNG HTTP error: {e}")
                if attempt < config.max_retries - 1:
                    continue
            except Exception as e:
                logger.warning(f"SearXNG error: {e}")
                break

        return []

    async def _duckduckgo_search(
        self,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """
        Search using DuckDuckGo.

        Fallback when SearXNG instances fail.
        """
        results = []
        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))
                for r in search_results:
                    results.append(SearchResult(
                        title=r.get('title', 'No title'),
                        url=r.get('href', ''),
                        snippet=r.get('body', ''),
                        source_type="duckduckgo"
                    ))
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")

        return results

    async def _wikipedia_search(self, query: str) -> List[SearchResult]:
        """
        Search Wikipedia using LangChain's WikipediaAPIWrapper.
        """
        results = []
        try:
            wiki = WikipediaAPIWrapper(top_k_results=3)
            # Run synchronous wiki search in thread pool
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, wiki.run, query)

            if summary and not summary.startswith("No good Wikipedia Search"):
                results.append(SearchResult(
                    title=f"Wikipedia: {query}",
                    url="https://wikipedia.org",
                    snippet=summary[:2000],
                    source_type="wikipedia",
                    credibility_score=0.9  # Wikipedia is generally credible
                ))
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")

        return results

    def _format_results(self, results: List[SearchResult]) -> str:
        """Format search results as markdown."""
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"**{i}. {r.title}**\n"
                f"[{r.url}]({r.url})\n"
                f"{r.snippet}\n"
            )
        return "\n\n---\n\n".join(formatted)


# =============================================================================
# Wikipedia Tool
# =============================================================================

class WikipediaTool(BaseTool):
    """
    Wikipedia search tool using LangChain's WikipediaAPIWrapper.

    Provides encyclopedic, well-sourced information on established topics.
    """

    name: str = "wikipedia_search"
    description: str = """Search Wikipedia for encyclopedic information.

    Use this tool when:
    - Need factual, well-sourced information on established topics
    - Getting background/context on concepts, people, places
    - Finding related topics through Wikipedia's link structure

    Args:
        query: The topic or concept to search for
        top_k: Number of Wikipedia pages to return (1-10, default 3)

    Returns:
        List of page summaries with title, url, and content

    Tips:
        - Works best for established, well-documented topics
        - May not have info on very recent events or niche topics
        - Good starting point before web_search for current info

    Examples:
        GOOD: "Knowledge graph"
        GOOD: "Transformer architecture (machine learning)"
        GOOD: "Python programming language history"
        AVOID: Very recent news or obscure topics
    """
    args_schema: Type[BaseModel] = WikipediaSearchInput

    top_k_results: int = 3
    doc_content_chars_max: int = 4000
    lang: str = "en"

    def _run(self, query: str, top_k: int = 3) -> str:
        """Execute Wikipedia search."""
        wiki = WikipediaAPIWrapper(
            top_k_results=top_k,
            lang=self.lang,
            doc_content_chars_max=self.doc_content_chars_max
        )
        try:
            result = wiki.run(query)
            if result and not result.startswith("No good Wikipedia Search"):
                return result
            return f"No Wikipedia results found for: {query}"
        except Exception as e:
            return f"Wikipedia search error: {str(e)}\n\nTry web_search instead."


# =============================================================================
# Tool Factory
# =============================================================================

def create_search_tools(config: Optional[DeepResearchConfig] = None) -> Dict[str, BaseTool]:
    """
    Create search tool instances.

    Args:
        config: Optional configuration. Uses env vars if not provided.

    Returns:
        Dictionary of tool name -> tool instance
    """
    config = config or DeepResearchConfig.from_env()

    return {
        "web_search": ResilientSearchTool(config=config),
        "wikipedia_search": WikipediaTool(top_k_results=config.max_search_results),
    }