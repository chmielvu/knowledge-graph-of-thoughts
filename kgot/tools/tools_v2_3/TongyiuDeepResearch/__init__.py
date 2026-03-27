"""
Tongyiu Deep Research Subagent

A research agent that conducts deep multi-source research using:
- Tongyi-DeepResearch-30B-A3B (orchestrator via nanoGPT API)
- Mistral Medium 2505 (extraction LLM via Crawl4AI LLMExtractionStrategy)
- Crawl4AI (web crawling with LLM strategies - adapted from unclecode/crawl4ai)
- SearXNG (web search with fallback to DuckDuckGo)
- Pollinations Gemini Search (AI-powered web search)
- Wikipedia (encyclopedic search via LangChain)

Configuration:
    NANOGPT_API_KEY: API key for nanoGPT (Tongyi orchestrator)
    MISTRAL_API_KEY: API key for Mistral (extraction LLM)
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type

import httpx
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
    LLMExtractionStrategy,
)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from duckduckgo_search import DDGS
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import new components
from .configuration import DeepResearchConfig, SearchAPI
from .state import ResearchState, Source, ResearchGap
from .tools import ResilientSearchTool, WikipediaTool, create_search_tools
from .agent import create_research_agent, run_research


# =============================================================================
# Output Contracts (Pydantic Models)
# =============================================================================

class Source(BaseModel):
    """A research source consulted during investigation."""
    url: str = Field(description="URL of the source")
    title: str = Field(description="Title of the source page")
    relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relevance score 0-1 indicating how relevant this source was"
    )
    excerpt: Optional[str] = Field(
        default=None,
        description="Key excerpt from the source"
    )
    accessed_at: datetime = Field(
        default_factory=datetime.now,
        description="When this source was accessed"
    )


class KeyFinding(BaseModel):
    """A key finding from the research."""
    statement: str = Field(description="The finding statement")
    supporting_sources: List[str] = Field(
        default_factory=list,
        description="URLs of sources supporting this finding"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this specific finding"
    )


class ResearchGap(BaseModel):
    """An area that needs more research."""
    topic: str = Field(description="The topic needing more research")
    reason: str = Field(description="Why this area needs more investigation")


class ResearchOutput(BaseModel):
    """Structured output from deep research."""
    answer: str = Field(description="The synthesized answer to the research query")
    key_findings: List[KeyFinding] = Field(
        default_factory=list,
        description="Key findings discovered during research"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="All sources consulted during research"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the research findings"
    )
    gaps: Optional[List[ResearchGap]] = Field(
        default=None,
        description="Areas that need more research"
    )
    research_depth_used: Literal["quick", "standard", "deep"] = Field(
        default="standard",
        description="The depth level used for this research"
    )
    total_sources_consulted: int = Field(
        default=0,
        description="Total number of unique sources consulted"
    )
    research_duration_seconds: float = Field(
        default=0.0,
        description="Time taken to complete research in seconds"
    )


class ResearchInput(BaseModel):
    """Input for the deep research tool."""
    query: str = Field(description="The research question or topic to investigate")
    depth: Literal["quick", "standard", "deep"] = Field(
        default="standard",
        description="Research depth: quick (1-2 sources), standard (3-5 sources), deep (5-10 sources)"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Optional specific aspects to focus research on"
    )
    exclude_sources: Optional[List[str]] = Field(
        default=None,
        description="Optional domains/URLs to exclude from research"
    )
    require_sources: Optional[int] = Field(
        default=None,
        description="Minimum number of sources required"
    )


# =============================================================================
# Tool Input Schemas
# =============================================================================

class WebSearchInput(BaseModel):
    """Input for web search tool."""
    queries: List[str] = Field(
        description="List of search queries to execute",
        min_length=1
    )
    max_results_per_query: int = Field(
        default=5,
        description="Maximum results per query"
    )


class VisitInput(BaseModel):
    """Input for visit tool (Alibaba-inspired)."""
    urls: List[str] = Field(
        description="URL(s) to visit and extract information from",
        min_length=1
    )
    goal: str = Field(
        description="The specific information goal for visiting these URLs"
    )


class DeepCrawlInput(BaseModel):
    """Input for deep crawl tool."""
    url: str = Field(description="Starting URL to crawl")
    max_depth: int = Field(default=2, description="Maximum crawl depth")
    max_pages: int = Field(default=20, description="Maximum pages to crawl")


class LLMExtractInput(BaseModel):
    """Input for LLM extraction tool."""
    url: str = Field(description="URL to extract data from")
    extraction_schema: Dict[str, Any] = Field(
        description="JSON schema describing fields to extract"
    )
    instruction: str = Field(
        description="Natural language instruction for extraction"
    )


class MarkdownFetchInput(BaseModel):
    """Input for markdown fetch tool."""
    url: str = Field(description="URL to fetch")
    css_selector: Optional[str] = Field(
        default=None,
        description="Optional CSS selector to focus extraction"
    )


class AIWebSearchInput(BaseModel):
    """Input for AI web search tool (Pollinations Gemini Search)."""
    query: str = Field(description="The search query to investigate")
    detailed: bool = Field(
        default=False,
        description="Whether to get comprehensive answer with more details"
    )


# =============================================================================
# System Prompt (Inspired by Alibaba DeepResearch)
# =============================================================================

TONGYIU_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Research Methodology

1. **Discovery Phase**: Start with web_search to identify relevant sources. Use multiple queries to get diverse perspectives. For complex topics, also use ai_web_search for AI-powered search with better context understanding.

2. **Deep Dive Phase**: Use visit to extract detailed information from promising URLs. Always provide a clear goal for what you're looking for.

3. **Verification Phase**: Cross-reference findings across multiple sources. Look for consensus and contradictions.

4. **Extraction Phase**: Use llm_extract for structured data extraction when you need specific fields from pages.

5. **Synthesis Phase**: Combine findings into a coherent, well-structured response internally.

# Quality Standards

- Always cite sources for factual claims using [source: URL] format
- Distinguish between verified facts and claims needing verification
- Note contradictions between sources and explain discrepancies
- Assess source credibility and recency
- Identify knowledge gaps when present
- Provide confidence assessments for key claims

# Tools

You have access to the following tools:

## web_search
Perform web searches and return results with titles, snippets, and URLs.
Input: List of search queries
Use for initial discovery and finding relevant sources.

## ai_web_search
AI-powered web search using Pollinations Gemini Search.
Input: A search query
Returns: Comprehensive answer with web sources
Use when you need contextual understanding or the topic is complex.

## visit
Visit URLs and extract information relevant to a specific goal.
Input: List of URLs and a goal describing what information to extract
Use to get detailed content from specific pages.

## deep_crawl
Crawl a website multiple levels deep, discovering and extracting linked pages.
Input: Starting URL, max depth, max pages
Use for comprehensive coverage of documentation sites or blogs.

## llm_extract
Extract structured data from a webpage using natural language instructions.
Input: URL, JSON schema for output, extraction instruction
Use when you need specific structured data from a page.

# Output Format

When research is complete:
1. Provide your synthesized answer in <answer></answer> tags
2. List key findings with source citations
3. Provide overall confidence assessment
4. Note any research gaps identified

Current date: {current_date}"""


# =============================================================================
# Configuration
# =============================================================================

class TongyiuConfig(BaseModel):
    """Configuration for Tongyiu Deep Research Agent."""

    # nanoGPT API for Tongyi orchestrator
    nanogpt_api_key: str = Field(default="", description="nanoGPT API key for Tongyi")
    tongyi_model: str = Field(
        default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        description="Tongyi model identifier"
    )

    # Mistral API for extraction
    mistral_api_key: str = Field(default="", description="Mistral API key")
    mistral_model: str = Field(
        default="mistral-medium-2505",
        description="Mistral model for extraction"
    )

    # Pollinations API for AI web search (optional)
    pollinations_api_key: str = Field(
        default="",
        description="Pollinations API key for AI-powered web search (optional for free tier)"
    )

    # Research settings
    max_search_results: int = Field(default=5, description="Max results per search query")
    default_crawl_depth: int = Field(default=2, description="Default crawl depth")
    default_max_pages: int = Field(default=20, description="Default max pages per crawl")

    # API endpoints
    nanogpt_base_url: str = Field(
        default="https://nano-gpt.com/api/subscription/v1",
        description="nanoGPT API base URL"
    )
    pollinations_api_base: str = Field(
        default="https://text.pollinations.ai",
        description="Pollinations API base URL"
    )

    @classmethod
    def from_env(cls) -> "TongyiuConfig":
        """Create config from environment variables."""
        return cls(
            nanogpt_api_key=os.getenv("NANOGPT_API_KEY", ""),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
            pollinations_api_key=os.getenv("POLLINATIONS_API_KEY", ""),
        )


# =============================================================================
# Tools
# =============================================================================

class WebSearchTool(BaseTool):
    """Web search tool using DuckDuckGo."""

    name: str = "web_search"
    description: str = "Perform web searches to find relevant sources. Returns results with titles, snippets, and URLs."
    args_schema: Type[BaseModel] = WebSearchInput

    max_results: int = 5

    def _run(self, queries: List[str], max_results_per_query: int = 5) -> str:
        return asyncio.run(self._arun(queries, max_results_per_query))

    async def _arun(self, queries: List[str], max_results_per_query: int = 5) -> str:
        results = []
        with DDGS() as ddgs:
            for query in queries:
                search_results = list(ddgs.text(query, max_results=max_results_per_query))
                for r in search_results:
                    results.append(
                        f"**{r.get('title', 'No title')}**\n"
                        f"{r.get('body', '')}\n"
                        f"URL: {r.get('href', '')}"
                    )

        return "\n\n---\n\n".join(results) if results else "No results found"


class VisitTool(BaseTool):
    """
    Visit URLs and extract information (Alibaba-inspired).

    Uses Crawl4AI with PruningContentFilter for clean, focused content extraction.
    Adapted from crawl4ai patterns.
    """

    name: str = "visit"
    description: str = "Visit webpage(s) and extract information relevant to a specific goal. Returns clean, filtered content."
    args_schema: Type[BaseModel] = VisitInput

    def _run(self, urls: List[str], goal: str) -> str:
        return asyncio.run(self._arun(urls, goal))

    async def _arun(self, urls: List[str], goal: str) -> str:
        """
        Visit URLs with goal-oriented extraction.

        Uses Crawl4AI's PruningContentFilter to extract the most relevant content,
        filtering out navigation, ads, and boilerplate.
        """
        results = []

        # Crawl4AI config with content filtering (adapted from quickstart patterns)
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            excluded_tags=["nav", "footer", "aside", "advertisement", "script", "style"],
            remove_overlay_elements=True,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,
                    threshold_type="fixed",
                    min_word_threshold=0
                ),
                options={"ignore_links": True},
            ),
            verbose=False
        )

        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url, config=config)
                    if result.success and result.markdown:
                        # Use fit_markdown if available for cleaner content
                        content = result.markdown[:4000]
                        results.append(f"## Source: {url}\n\n{content}")
                    else:
                        results.append(f"## Source: {url}\n\nFailed to extract content")
                except Exception as e:
                    results.append(f"## Source: {url}\n\nError: {str(e)}")

        if not results:
            return "Failed to fetch any URLs"

        return "\n\n---\n\n".join(results)


class DeepCrawlTool(BaseTool):
    """
    Deep crawl tool for multi-level website exploration.

    Uses Crawl4AI's BFSDeepCrawlStrategy for breadth-first crawling.
    Adapted from crawl4ai deep crawling patterns.
    """

    name: str = "deep_crawl"
    description: str = "Crawl a website multiple levels deep, discovering and extracting content from linked pages."
    args_schema: Type[BaseModel] = DeepCrawlInput

    def _run(self, url: str, max_depth: int = 2, max_pages: int = 20) -> str:
        return asyncio.run(self._arun(url, max_depth, max_pages))

    async def _arun(self, url: str, max_depth: int = 2, max_pages: int = 20) -> str:
        """
        Perform deep crawl using BFS strategy.

        Adapted from crawl4ai BFSDeepCrawlStrategy implementation.
        """
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=max_depth,
                include_external=False,
                max_pages=max_pages
            ),
            excluded_tags=["nav", "footer", "aside", "advertisement"],
            remove_overlay_elements=True,
            verbose=False
        )

        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun(url, config=config)

            aggregated = []
            for result in results:
                if result.success and result.markdown:
                    aggregated.append(f"## Source: {result.url}\n\n{result.markdown[:1500]}")

            return "\n\n---\n\n".join(aggregated) if aggregated else "No content extracted"


class LLMExtractTool(BaseTool):
    """LLM-powered structured extraction from web pages."""

    name: str = "llm_extract"
    description: str = "Extract structured data from a webpage using natural language instructions and a JSON schema."
    args_schema: Type[BaseModel] = LLMExtractInput

    mistral_api_key: str = ""
    mistral_model: str = "mistral-medium-2505"

    def _run(self, url: str, extraction_schema: Dict[str, Any], instruction: str) -> str:
        return asyncio.run(self._arun(url, extraction_schema, instruction))

    async def _arun(self, url: str, extraction_schema: Dict[str, Any], instruction: str) -> str:
        llm_config = LLMConfig(
            provider=f"mistral/{self.mistral_model}",
            api_token=self.mistral_api_key
        )

        strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=extraction_schema,
            extraction_type="schema",
            instruction=instruction,
            extra_args={"temperature": 0.1}
        )

        config = CrawlerRunConfig(extraction_strategy=strategy)

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url, config=config)

            if result.success and result.extracted_content:
                return result.extracted_content
            return f"Extraction failed: {result.error_message or 'Unknown error'}"


class AIWebSearchTool(BaseTool):
    """
    AI-powered web search tool using Pollinations Gemini Search.

    This tool performs web searches with AI-powered contextual understanding,
    returning comprehensive answers with source citations.

    Adapted from Pollinations.ai API patterns.
    """

    name: str = "ai_web_search"
    description: str = (
        "AI-powered web search that returns comprehensive answers with sources. "
        "Use when you need contextual understanding or the topic is complex."
    )
    args_schema: Type[BaseModel] = AIWebSearchInput

    api_base: str = "https://text.pollinations.ai"

    def _run(self, query: str, detailed: bool = False) -> str:
        return asyncio.run(self._arun(query, detailed))

    async def _arun(self, query: str, detailed: bool = False) -> str:
        """
        Perform AI-powered web search.

        Uses gemini-search model which combines Google Search with Gemini
        for contextual, sourced answers.
        """
        payload = {
            "model": "gemini-search",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant. Provide comprehensive, accurate answers with source citations."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_base,
                json=payload,
                timeout=60.0
            )

            if response.status_code == 200:
                return response.text
            return f"Search API Error: {response.status_code}"


# =============================================================================
# Tongyiu Deep Research Tool
# =============================================================================

class TongyiuDeepResearchTool(BaseTool):
    """
    Deep Research Agent - Conducts comprehensive multi-source research.

    Uses Tongyi-DeepResearch-30B-A3B for orchestration and
    Mistral Medium 2505 for extraction tasks.

    Returns structured research output with:
    - Synthesized answer
    - Key findings with citations
    - Source list
    - Confidence assessment
    - Research gaps
    """

    name: str = "deep_research_tool"
    description: str = """
Comprehensive multi-source research agent. Use when you need THOROUGH investigation.

This tool searches MULTIPLE sources, crawls web pages, and synthesizes findings
into a structured report. It will:
1. Generate optimized search queries
2. Search SearXNG/DuckDuckGo and optionally Pollinations AI search
3. Crawl promising pages with Crawl4AI
4. Extract structured information using LLM strategies
5. Synthesize findings into a comprehensive answer

Use for:
- Market research and competitor analysis (set depth="deep")
- Technical documentation synthesis across multiple sources
- Academic literature review
- Fact-checking requiring multiple sources
- Any question requiring comprehensive research

Parameters:
- query: The research question or topic
- depth: "quick" (1-2 sources), "standard" (3-5 sources), "deep" (5-10 sources)
- focus_areas: Optional list of aspects to focus on
- exclude_sources: URLs to skip

Returns: Structured output with answer, key_findings, sources, confidence, gaps.

AVOID for:
- Simple lookups (use LLM knowledge)
- Single URL navigation (use browser_use_tool)
- Quick fact checks (use pollinations_search)
"""

    args_schema: Type[BaseModel] = ResearchInput

    config: Optional[TongyiuConfig] = None

    def __init__(self, config: Optional[TongyiuConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or TongyiuConfig.from_env()

    def _run(
        self,
        query: str,
        depth: Literal["quick", "standard", "deep"] = "standard",
        focus_areas: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        require_sources: Optional[int] = None
    ) -> ResearchOutput:
        return asyncio.run(self._arun(
            query, depth, focus_areas, exclude_sources, require_sources
        ))

    async def _arun(
        self,
        query: str,
        depth: Literal["quick", "standard", "deep"] = "standard",
        focus_areas: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        require_sources: Optional[int] = None
    ) -> ResearchOutput:
        start_time = time.time()

        # Depth configuration
        depth_config = {
            "quick": {"max_iterations": 5, "min_sources": 1, "max_sources": 2},
            "standard": {"max_iterations": 10, "min_sources": 3, "max_sources": 5},
            "deep": {"max_iterations": 20, "min_sources": 5, "max_sources": 10}
        }
        depth_settings = depth_config.get(depth, depth_config["standard"])

        # Initialize tools
        tools = self._create_tools()

        # Run research loop
        sources: List[Source] = []
        findings: List[KeyFinding] = []
        all_content = []

        # Phase 1: Discovery - Use both regular search and AI-powered search
        search_tool = tools["web_search"]
        ai_search_tool = tools["ai_web_search"]

        # Build search queries
        search_queries = [query]
        if focus_areas:
            search_queries.extend([f"{query} {area}" for area in focus_areas])

        # Regular web search
        search_results = await search_tool._arun(search_queries, self.config.max_search_results)
        all_content.append(f"## Web Search Results\n\n{search_results}")

        # AI-powered web search for deeper context
        ai_search_result = await ai_search_tool._arun(query=query)
        all_content.append(f"## AI-Powered Search Results\n\n{ai_search_result}")

        # Phase 2: Deep dive (visit top URLs from regular search)
        visit_tool = tools["visit"]
        urls = self._extract_urls(search_results, exclude_sources)[:depth_settings["max_sources"]]

        if urls:
            visit_results = await visit_tool._arun(urls=urls, goal=query)
            all_content.append(f"## Visited Pages\n\n{visit_results}")

            for url in urls:
                sources.append(Source(
                    url=url,
                    title=f"Source from research",
                    relevance=0.8
                ))

        # Phase 3: Synthesis using Tongyi orchestrator via nanoGPT
        combined_content = "\n\n---\n\n".join(all_content)

        # Tongyi-DeepResearch-30B-A3B via nanoGPT (OpenAI-compatible)
        orchestrator = ChatOpenAI(
            model=self.config.tongyi_model,
            api_key=self.config.nanogpt_api_key,
            base_url=self.config.nanogpt_base_url,
            temperature=0.1,
            max_retries=3
        )

        synthesis_prompt = f"""Based on the following research data, provide a comprehensive answer to: {query}

Research Data:
{combined_content[:12000]}

Please provide:
1. A synthesized answer with source citations
2. Key findings (bullet points)
3. Overall confidence level (0-1)
4. Any research gaps identified

Format your response with clear sections."""

        synthesis = await orchestrator.ainvoke(synthesis_prompt)
        answer = synthesis.content if hasattr(synthesis, 'content') else str(synthesis)

        # Build output
        duration = time.time() - start_time

        output = ResearchOutput(
            answer=answer,
            key_findings=findings,
            sources=sources,
            confidence=0.7,
            research_depth_used=depth,
            total_sources_consulted=len(sources),
            research_duration_seconds=duration
        )

        return output

    def _create_tools(self) -> Dict[str, BaseTool]:
        """Create tool instances."""
        return {
            "web_search": WebSearchTool(max_results=self.config.max_search_results),
            "ai_web_search": AIWebSearchTool(api_base=self.config.pollinations_api_base),
            "visit": VisitTool(),
            "deep_crawl": DeepCrawlTool(),
            "llm_extract": LLMExtractTool(
                mistral_api_key=self.config.mistral_api_key,
                mistral_model=self.config.mistral_model
            ),
        }

    def _extract_urls(self, search_results: str, exclude: Optional[List[str]] = None) -> List[str]:
        """Extract URLs from search results."""
        import re
        urls = re.findall(r'URL:\s*(https?://[^\s\n]+)', search_results)

        if exclude:
            urls = [u for u in urls if not any(ex in u for ex in exclude)]

        return urls


# =============================================================================
# Convenience Functions
# =============================================================================

def research(
    query: str,
    depth: Literal["quick", "standard", "deep"] = "standard",
    config: Optional[TongyiuConfig] = None
) -> ResearchOutput:
    """
    Convenience function for quick research.

    Args:
        query: Research question or topic.
        depth: 'quick', 'standard', or 'deep'.
        config: Optional configuration.

    Returns:
        Structured research output.
    """
    tool = TongyiuDeepResearchTool(config=config)
    return tool.invoke({"query": query, "depth": depth})


async def aresearch(
    query: str,
    depth: Literal["quick", "standard", "deep"] = "standard",
    config: Optional[TongyiuConfig] = None
) -> ResearchOutput:
    """
    Async convenience function for research.

    Args:
        query: Research question or topic.
        depth: 'quick', 'standard', or 'deep'.
        config: Optional configuration.

    Returns:
        Structured research output.
    """
    tool = TongyiuDeepResearchTool(config=config)
    return await tool._arun(query, depth)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "TongyiuConfig",
    "DeepResearchConfig",
    "SearchAPI",
    # Input/Output
    "ResearchInput",
    "ResearchOutput",
    "Source",
    "KeyFinding",
    "ResearchGap",
    "ResearchState",
    # Tools
    "TongyiuDeepResearchTool",
    "WebSearchTool",
    "VisitTool",
    "DeepCrawlTool",
    "LLMExtractTool",
    "AIWebSearchTool",
    "ResilientSearchTool",
    "WikipediaTool",
    # Agent
    "create_research_agent",
    "run_research",
    # Functions
    "research",
    "aresearch",
]