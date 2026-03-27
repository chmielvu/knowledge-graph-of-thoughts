# Deep Research Agent Enhancement Plan

## Context

The current TongyiDeepResearchTool implementation has:
- ✅ Working Tongyi orchestrator via nanoGPT API
- ✅ Pydantic input/output contracts
- ⚠️ Rate-limited DuckDuckGo search
- ⚠️ Pollinations AI search requires API key
- ❌ No autonomous agent loop (linear pipeline only)
- ❌ No tool calling - Tongyi has tool descriptions but can't invoke them

**Goal**: Transform from linear pipeline to autonomous multi-tool agent with proper tool calling.

---

## Research Findings

### 1. Best Reference: `tarun7r/deep-research-agent` (147 ⭐)

**Architecture**: Multi-agent LangGraph system with:
- `ResearchPlanner` - Creates structured research plans
- `ResearchSearcher` - Executes searches with credibility scoring
- `ResearchSynthesizer` - Synthesizes findings
- `ReportWriter` - Generates final reports

**Tool Design Patterns** (excellent documentation):
```python
@tool
async def web_search(query: str, max_results: int = None) -> List[dict]:
    """Search the web for authoritative information...

    - Gathering factual information on any topic
    - Finding authoritative sources (academic, government, official docs)

    Args:
        query: A well-crafted search query string...

    Returns:
        List of dictionaries with title, url, snippet

    Tips:
        - Check URL domains: .edu, .gov, .org often indicate credibility
    """
```

**Key insight**: Tool descriptions include usage patterns, examples, AND tips for the LLM.

### 2. SearXNG on Hugging Face Spaces - Detailed Analysis

#### Existing HF Spaces Analyzed

**MaximusAI/SearXNG** (8 likes, Docker SDK):
```dockerfile
FROM searxng/searxng:latest
ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
ENV SEARXNG_BASE_URL="/"
ENV UWSGI_HTTP=0.0.0.0:7860
ENV GRANIAN_HOST=0.0.0.0
ENV GRANIAN_PORT=7860
COPY searxng/settings.yml /etc/searxng/settings.yml
EXPOSE 7860
CMD ["searxng"]
```

**senku21230/my-searxng-engine** (1 like, Docker SDK):
- Similar simple setup with `chmod 777 /etc/searxng` for permissions
- Cache-busting date comment for forcing rebuilds

**Key Finding**: Both spaces run **standalone SearXNG without Redis/Valkey**. This is acceptable for HF Spaces where caching isn't critical.

#### settings.yml Configurations Found

**MaximusAI's settings.yml**:
```yaml
use_default_settings: true
server:
  secret_key: "rnd_searxng_secret_2026_x7k9"
  bind_address: "0.0.0.0"
  port: 7860
  image_proxy: false
  real_ip_header: "X-Forwarded-For"
  real_ip_proxies: ["0.0.0.0/0"]
search:
  safe_search: 0
  formats: [html, json]
```

#### Official SearXNG Docker Setup (searxng/searxng-docker)

Full production setup with Redis caching:
```yaml
services:
  searxng:
    image: docker.io/searxng/searxng:latest
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      - SEARXNG_BASE_URL=https://${SEARXNG_HOSTNAME:-localhost}/
  redis:
    image: docker.io/valkey/valkey:8-alpine
```

#### User's Existing Space: `chmielvu/web-search-mcp`

**Current Structure**:
- SDK: **Gradio** (not Docker)
- Already has `SearxNGInterfaceWrapper` but **incomplete** - missing `searxng_host` input
- 6 search engines: Brave, DuckDuckGo, SearxNG, SerpAPI, Serper, Tavily

**Critical Issue**: SearxNGInterfaceWrapper requires `searxng_host` parameter but doesn't expose it in inputs:
```python
# Current (broken) - searxng_host not provided
async def searxng_search(query: str, searxng_host: str, max_results: int = 5):
    ...

class SearxNGInterfaceWrapper(BaseInterfaceWrapper):
    def __init__(self):
        super().__init__(
            fn=searxng_search,  # Requires searxng_host!
            inputs=[
                gr.Textbox(label="Search Query"),  # Only query provided
                gr.Slider(minimum=1, maximum=10, step=1, value=5),
            ],
        )
```

#### Downloaded Implementations (app (9).py, app (10).py)

**Both files are identical** - Gradio apps that:
- Call **external** SearXNG instances (not self-hosted)
- Support content extraction via BeautifulSoup or Trafilatura
- Include LLM summarization via HuggingFace InferenceClient (Mistral-Nemo)

**Useful patterns to port**:
1. User-agent rotation for anti-bot bypass
2. Exponential backoff retry for rate limits (429 errors)
3. Dual extraction methods (BS4 vs Trafilatura)
4. JSON API format with advanced params

#### Integration Constraints & Options

**Cannot add SearXNG to existing web-search-mcp Space** because Gradio SDK doesn't support multi-container.

**Options**:
1. **Create separate SearXNG Space** (Recommended) - Use MaximusAI's Dockerfile as template
2. **Fix existing SearxNGInterfaceWrapper** - Add `searxng_host` input or hardcode a public instance
3. **Use public SearXNG instances** - `https://searx.be`, `https://search.bus-hit.me`

### 3. Crawl4AI Best Practices (v0.8.5)

Key patterns to adopt:
- `BFSDeepCrawlStrategy` for multi-page crawling
- `LLMExtractionStrategy` with schema-based extraction
- `PruningContentFilter` for noise removal
- Anti-bot detection with proxy escalation
- Prefetch mode for 5-10x faster URL discovery

---

## Recommended Tool Set for Deep Research Agent

### Core Tools (5 tools - avoid "too many tools" anti-pattern)

| Tool | Purpose | Implementation |
|------|---------|----------------|
| `web_search` | Find sources | SearXNG API (fallback to DDG) |
| `visit_page` | Extract content | Crawl4AI with PruningContentFilter |
| `deep_crawl` | Multi-page sites | Crawl4AI BFSDeepCrawlStrategy |
| `extract_structured` | Get specific data | Crawl4AI LLMExtractionStrategy |
| `verify_claims` | Cross-reference | Multi-source comparison |

### Tool Schema Design Principles

1. **Explicit error messages** that help LLM recover:
```python
except RateLimitError:
    return "Rate limited. Try: 1) Wait 60s 2) Use different search terms 3) Try ai_web_search"
```

2. **Usage examples in description**:
```python
"""
Good examples:
- "WebSocket vs HTTP streaming performance comparison"
- "site:arxiv.org large language models"

Avoid:
- Single words: "AI", "cloud"
- Overly long queries (>15 words)
"""
```

3. **Return structured data** with metadata:
```python
return {
    "success": True,
    "content": extracted_text,
    "source_url": url,
    "word_count": len(extracted_text.split()),
    "extraction_method": "pruning_filter"
}
```

---

## Implementation Plan

### Phase 1: Add SearXNG Search Tool

**Files to modify**:
- `kgot/tools/tools_v2_3/TongyiuDeepResearch/__init__.py`

Add SearXNG integration:
```python
class SearXNGSearchTool(BaseTool):
    name = "web_search"
    description = """Search the web using SearXNG metasearch...

    Args:
        query: Search query (be specific, include context)

    Returns:
        List of results with title, url, snippet

    Tips:
        - Use "site:edu" for academic sources
        - Add year for current info: "AI trends 2025"
    """

    def _run(self, query: str) -> List[dict]:
        # Try SearXNG first, fallback to DDG
        ...
```

**Configuration**:
```python
# In TongyiuConfig
searxng_url: str = Field(
    default="https://saneowl-searxng-search-engine.hf.space",
    description="SearXNG API endpoint"
)
```

### Phase 2: Improve Tool Descriptions

Rewrite tool docstrings following `tarun7r/deep-research-agent` patterns:
- Add "When to use" section
- Add "Good examples" and "Avoid" examples
- Add "Returns" with field descriptions
- Add "Tips" section

### Phase 3: Add LangGraph Agent Loop

**New file**: `kgot/tools/tools_v2_3/TongyiuDeepResearch/agent.py`

```python
from langgraph.graph import StateGraph, END

def create_research_agent():
    workflow = StateGraph(ResearchState)

    # Nodes
    workflow.add_node("plan", plan_research)
    workflow.add_node("search", execute_search)
    workflow.add_node("extract", extract_content)
    workflow.add_node("synthesize", synthesize_findings)

    # Edges
    workflow.add_edge("plan", "search")
    workflow.add_conditional_edges("search", should_continue, {
        "search": "search",  # More searches needed
        "extract": "extract"
    })
    ...
```

---

## SearXNG HF Space Integration - Recommended Approach

### Recommended: Create Dedicated SearXNG Space

**Dockerfile** (based on MaximusAI/SearXNG):
```dockerfile
FROM searxng/searxng:latest
ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
ENV SEARXNG_BASE_URL="/"
ENV UWSGI_HTTP=0.0.0.0:7860
ENV GRANIAN_HOST=0.0.0.0
ENV GRANIAN_PORT=7860
COPY searxng/settings.yml /etc/searxng/settings.yml
EXPOSE 7860
CMD ["searxng"]
```

**searxng/settings.yml**:
```yaml
use_default_settings: true
server:
  secret_key: "your_random_secret_key_here"
  bind_address: "0.0.0.0"
  port: 7860
  limiter: false  # Disable for public API use
  image_proxy: false
search:
  safe_search: 0
  formats: [html, json]
```

**API Endpoint**: `https://your-searxng-space.hf.space/search?q=query&format=json`

### Fix Existing web-search-mcp SearxNGInterfaceWrapper

Add the missing `searxng_host` input to `search_engines/searxng.py`:
```python
class SearxNGInterfaceWrapper(BaseInterfaceWrapper):
    def __init__(self):
        super().__init__(
            fn=searxng_search,
            inputs=[
                gr.Textbox(label="Search Query"),
                gr.Textbox(label="SearXNG Host", value="https://searx.be"),  # ADD THIS
                gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Results"),
            ],
        )
```

---

## Verification Plan

```bash
# Test SearXNG tool
python -c "
from kgot.tools.tools_v2_3.TongyiuDeepResearch import SearXNGSearchTool
tool = SearXNGSearchTool()
results = tool._run('Python async programming tutorial')
print(f'Found {len(results)} results')
print(results[0]['title'])
"

# Test full agent
python -c "
from kgot.tools.tools_v2_3.TongyiuDeepResearch import TongyiuDeepResearchTool
tool = TongyiuDeepResearchTool()
result = tool._run(query='What is RAG in AI?', depth='quick')
print(result.answer[:500])
"
```

---

## Files to Modify/Create

| File | Action | Priority |
|------|--------|----------|
| `TongyiuDeepResearch/tools.py` | Create new file with ResilientSearchTool, WikipediaTool | HIGH |
| `TongyiuDeepResearch/configuration.py` | Create DeepResearchConfig with search fallback chain | HIGH |
| `TongyiuDeepResearch/state.py` | Create ResearchState for LangGraph workflow | HIGH |
| `TongyiuDeepResearch/agent.py` | Create LangGraph agent with conditional edges | HIGH |
| `TongyiuDeepResearch/__init__.py` | Update tool definitions with best practice descriptions | HIGH |
| `chmielvu/web-search-mcp` (HF Space) | Fix SearxNGInterfaceWrapper - add searxng_host input | MEDIUM |
| New HF Space | Create dedicated SearXNG Space using MaximusAI template | MEDIUM |

---

## LangChain Reference Implementations Analyzed

### local-deep-researcher (langchain-ai/local-deep-researcher)

**Key patterns extracted**:
1. `Configuration` class with `from_runnable_config()` for runtime config
2. Tool calling vs JSON mode toggle
3. Multi-search API support (perplexity, tavily, duckduckgo, searxng)
4. `generate_query` -> `web_research` -> `summarize` -> `reflect` workflow
5. Conditional edges based on `research_loop_count`

**searxng_search implementation**:
```python
def searxng_search(query: str, max_results: int = 3, fetch_full_page: bool = False):
    host = os.environ.get("SEARXNG_URL", "http://localhost:8888")
    s = SearxSearchWrapper(searx_host=host)
    search_results = s.results(query, num_results=max_results)
    # Format and optionally fetch full page content
```

### LangChain WikipediaAPIWrapper

```python
class WikipediaAPIWrapper(BaseModel):
    top_k_results: int = 3
    lang: str = "en"
    doc_content_chars_max: int = 4000

    def run(self, query: str) -> str:
        page_titles = self.wiki_client.search(query[:300], results=self.top_k_results)
        summaries = [self._fetch_page(title) for title in page_titles]
        return "\n\n".join(summaries)[:self.doc_content_chars_max]
```

### LangChain SearxSearchWrapper

```python
class SearxSearchWrapper(BaseModel):
    searx_host: str  # Required
    engines: Optional[List[str]] = []
    categories: Optional[List[str]] = []
    k: int = 10  # Max results

    def results(self, query: str, num_results: int) -> List[Dict]:
        # Returns list of {snippet, title, link, engines, category}
```

---

## High-ROI Enhancement Recommendations

| Priority | Enhancement | Est. Effort | Impact |
|----------|------------|-------------|--------|
| **P0** | Install streamlit-flow and test graph visualization | 5 min | High - Visual debugging |
| **P1** | Wire FalkorDB semantic search into main controller | 2-3 hrs | High - Better retrieval |
| **P2** | Add SearXNG fallback chain to deep_research_tool | 1-2 hrs | Medium - Reliable search |
| **P3** | Improve tool descriptions following best practices | 1 hr | Medium - Better agent tool selection |
| **P4** | Add LangGraph agent loop for deep research | 4-6 hrs | High - Autonomous multi-step |

---

*Created: 2026-03-26*