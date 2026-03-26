# Tongyiu Deep Research Subagent

A LangChain-based research agent that conducts deep multi-source research using:
- **Orchestrator**: `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` (30B MoE, 3B active per token)
- **Extraction LLM**: `mistral-medium-2505` (Mistral Medium 3, May 2025)
- **Web Crawling**: Crawl4AI with LLM strategies (adapted from unclecode/crawl4ai)
- **Search**: DuckDuckGo + Pollinations Gemini Search
- **Synthesis**: Mistral Medium 2505 (placeholder for Tongyi integration)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Tongyiu Deep Research Agent                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │        Orchestrator: Tongyi-DeepResearch-30B-A3B          │      │
│  │        (via vLLM/HF Transformers)                          │      │
│  │        Research reasoning, tool selection, synthesis       │      │
│  └────────────────────────────────────────────────────────────┘      │
│                              │                                        │
│        ┌─────────────────────┼─────────────────────┐                 │
│        ▼                     ▼                     ▼                 │
│  ┌───────────┐        ┌───────────┐        ┌───────────┐            │
│  │web_search │        │   visit   │        │ai_web_search│           │
│  │(DuckDuckGo)│        │(Crawl4AI) │        │(Gemini Search)│         │
│  └───────────┘        └───────────┘        └───────────┘            │
│        │                     │                     │                 │
│        └─────────────────────┴─────────────────────┘                 │
│                              │                                        │
│        ┌─────────────────────┼─────────────────────┐                 │
│        ▼                     ▼                     ▼                 │
│  ┌───────────┐        ┌───────────┐        ┌───────────┐            │
│  │deep_crawl │        │llm_extract│        │ Synthesis │            │
│  │(BFS Crawl)│        │(Mistral)  │        │(Mistral)  │            │
│  └───────────┘        └───────────┘        └───────────┘            │
│                                                                       │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │               ResearchOutput (Pydantic Model)              │      │
│  │  • answer: str                                              │      │
│  │  • key_findings: List[KeyFinding]                          │      │
│  │  • sources: List[Source]                                    │      │
│  │  • confidence: float (0-1)                                  │      │
│  │  • gaps: List[ResearchGap]                                  │      │
│  └────────────────────────────────────────────────────────────┘      │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Tools (Adapted from Crawl4AI)

### 1. web_search
**Purpose**: Initial discovery via DuckDuckGo
- Input: List of search queries
- Returns: Results with titles, snippets, URLs

### 2. ai_web_search
**Purpose**: AI-powered web search with contextual understanding
- Input: Search query
- Uses: Pollinations `gemini-search` model (Google Search + Gemini)
- Returns: Comprehensive answer with sources

### 3. visit (Alibaba-inspired)
**Purpose**: Goal-oriented URL exploration
- Input: URLs + goal (what to extract)
- Uses: Crawl4AI AsyncWebCrawler with PruningContentFilter
- Returns: Clean, filtered markdown content

### 4. deep_crawl
**Purpose**: Multi-level website crawling
- Input: URL, max_depth, max_pages
- Uses: Crawl4AI BFSDeepCrawlStrategy
- Returns: Aggregated markdown from all pages

### 5. llm_extract
**Purpose**: Structured data extraction
- Input: URL, JSON schema, instruction
- Uses: Crawl4AI LLMExtractionStrategy with `mistral-medium-2505`

## Model Configuration

| Role | Model | Access |
|------|-------|--------|
| Orchestrator | `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` | vLLM/HF Transformers |
| Extraction | `mistral-medium-2505` | Mistral API |
| AI Web Search | `gemini-search` | Pollinations.ai |
| Synthesis | `mistral-medium-2505` | Mistral API |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MISTRAL_API_KEY` | Mistral API key for extraction |
| `TONGYIU_MODEL_PATH` | Local path to Tongyi model (optional) |
| `TONGYIU_DEVICE` | Device for Tongyi inference (default: "auto") |

## Crawl4AI Patterns Used

The implementation adapts these patterns from the Crawl4AI library:

### LLMExtractionStrategy
```python
from crawl4ai import LLMConfig, LLMExtractionStrategy

llm_config = LLMConfig(
    provider="mistral/mistral-medium-2505",
    api_token=os.environ["MISTRAL_API_KEY"]
)

strategy = LLMExtractionStrategy(
    llm_config=llm_config,
    schema=extraction_schema,
    extraction_type="schema",
    instruction="Extract specific fields...",
    extra_args={"temperature": 0.1}
)
```

### AsyncWebCrawler with Content Filtering
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    excluded_tags=["nav", "footer", "aside", "advertisement"],
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48)
    )
)

async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(url, config=config)
```

### BFSDeepCrawlStrategy
```python
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

config = CrawlerRunConfig(
    deep_crawl_strategy=BFSDeepCrawlStrategy(
        max_depth=2,
        include_external=False,
        max_pages=20
    )
)
```

## Research Methodology

Inspired by Alibaba DeepResearch:

1. **Discovery**: Multi-query web search for diverse sources
2. **AI Search**: Use gemini-search for contextual understanding
3. **Deep Dive**: Goal-oriented URL visiting with content filtering
4. **Verification**: Cross-reference across sources
5. **Synthesis**: Combine findings with source citations

## Usage

```python
from kgot.tools.tools_v2_3.TongyiuDeepResearch import (
    TongyiuDeepResearchTool,
    TongyiuConfig,
    research
)

# Quick usage
result = research("Latest advances in RAG systems", depth="deep")
print(result.answer)
print(f"Confidence: {result.confidence}")
print(f"Sources: {len(result.sources)}")

# With configuration
config = TongyiuConfig(
    mistral_api_key="your-key",
    max_search_results=10
)
tool = TongyiuDeepResearchTool(config=config)
result = tool.invoke({
    "query": "Compare LangGraph vs CrewAI",
    "depth": "standard",
    "focus_areas": ["architecture", "use cases", "performance"]
})
```

## Requirements

```
crawl4ai>=0.4.0
langchain>=0.3.0
langchain-mistralai>=0.2.0
duckduckgo-search>=6.0.0
httpx>=0.27.0
pydantic>=2.0.0
torch>=2.0.0
transformers>=4.40.0
```

## References

- [Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- [Tongyi-DeepResearch Model](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- [Mistral Medium 3](https://mistral.ai/news/mistral-medium-3/)
- [Crawl4AI](https://github.com/unclecode/crawl4ai)
- [Pollinations.ai](https://pollinations.ai)