# KGoT Enhancement Proposal: Tools, MCP Integration, and Vector Capabilities

**Date:** 2026-03-24
**Status:** Research Complete - Ready for Implementation Planning

---

## Executive Summary

This proposal outlines strategic enhancements for the Knowledge Graph of Thoughts (KGoT) framework across five key areas:

1. **Tool Upgrades** - Leverage HuggingFace ecosystem for enhanced capabilities
2. **MCP Server** - Expose KGoT tools via Model Context Protocol
3. **MCP Client** - Enable KGoT to consume external MCP tools
4. **Neo4j Vector Store** - Integrate vector search with graph traversal
5. **Additional Capabilities** - GraphRAG, hybrid retrieval, and tool dependency management

**Key Recommendation:** Prioritize MCP integration first (low effort, high impact), followed by Neo4j vector enhancement, then HuggingFace tool upgrades.

---

## 1. Tool Enhancement Opportunities

### 1.1 Current Tool Architecture

| Tool | Current Backend | Purpose |
|------|-----------------|---------|
| `ImageQuestionTool` | OpenAI VLM via LangChain | Vision QA |
| `TextInspectorTool` | Custom MarkdownConverter + LangChain | Document parsing |
| `SearchTool` (SurferTool) | Transformers ReactJsonAgent | Web search |
| `PollinationsSearchTool` | Pollinations API | AI-powered search |
| `RunPythonCodeTool` | Docker container | Code execution |
| `LangchainLLMTool` | LangChain | General LLM |
| `GraphVizTool` | - | Visualization |
| `ExtractZipTool` | - | Archive handling |

### 1.2 HuggingFace Replacement/Enhancement Matrix

| Priority | Current Tool | HuggingFace Alternative | Effort | Benefit |
|----------|--------------|------------------------|--------|---------|
| **High** | TextInspectorTool OCR | `mcp-tools/DeepSeek-OCR-experimental` | Low | Superior OCR, preserves structure |
| **High** | Speech transcription | `openai/whisper-large-v3` via Inference API | Low | Built-in audio support |
| **Medium** | ImageQuestionTool | Qwen2.5-VL-7B-Instruct | Medium | Open-source, strong VLM |
| **Medium** | Web Search | `smolagents.WebSearchTool` | Medium | Built-in agent integration |
| **Low** | Code Execution | `vmohan-sn/PythonCodeExec` MCP Space | Low | MCP-native alternative |

### 1.3 New Tool Capabilities

| New Tool | Source | Use Case |
|----------|--------|----------|
| Background Removal | `not-lain/background-removal` MCP Space | Image preprocessing |
| Image Segmentation | `prithivMLmods/SAM3-Image-Segmentation` | Object detection |
| Text-to-Speech | `ResembleAI/Chatterbox` MCP Space | Audio output |
| Image Generation | `mcp-tools/Qwen-Image` | Creative tasks |

### 1.4 Implementation Pattern: HFToolAdapter

```python
# Proposed: kgot/tools/tools_v2_3/HFToolAdapter.py

from langchain.tools import BaseTool
from huggingface_hub import InferenceClient
from typing import Optional

class HFInferenceTool(BaseTool):
    """Base class for HuggingFace Inference API tools"""
    name: str
    description: str
    model_id: str
    task: str
    api_token: Optional[str] = None

    def _run(self, **inputs):
        client = InferenceClient(model=self.model_id, token=self.api_token)
        return client.post(json=inputs, task=self.task)

class HFMCPTool(BaseTool):
    """Wrapper for HuggingFace MCP Spaces"""
    space_id: str
    mcp_url: str

    def _run(self, **kwargs):
        # Call MCP Space via SSE/HTTP
        ...
```

---

## 2. MCP Server Architecture (Exposing KGoT)

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Host (Claude Code)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │ MCP Protocol
┌─────────────────────────────▼───────────────────────────────────┐
│                     KGoT MCP Server                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    FastMCP Layer                             ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   ││
│  │  │ Tools    │ │Resources │ │ Prompts  │ │ Notifications│   ││
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────────────┘   ││
│  └───────┼────────────┼────────────┼──────────────────────────┘│
│  ┌───────▼────────────▼────────────▼──────────────────────────┐│
│  │                 Tool Manager                                ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Knowledge Graph (Neo4j/NetworkX/RDF4J)         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tool Exposure Strategy

**Expose all existing LangChain tools as MCP tools:**

```python
# Proposed: kgot/mcp/server.py
from fastmcp import FastMCP
from kgot.tools.tools_v2_3.tool_manager import ToolManager

mcp = FastMCP("KGoT Server")
tool_manager = ToolManager(usage_statistics=None)

# Auto-wrap LangChain tools
for lc_tool in tool_manager.get_tools():
    mcp.tool(
        name=lc_tool.name,
        description=lc_tool.description
    )(lc_tool.invoke)

# Expose knowledge graph as resource
@mcp.resource("kgot://graph/state")
async def get_graph_state() -> str:
    """Current knowledge graph state"""
    return tool_manager.graph.get_current_graph_state()

# Expose analysis prompts
@mcp.prompt()
def kgot_analysis_prompt(problem: str) -> str:
    """Template for knowledge graph analysis"""
    return f"Analyze this problem using knowledge graph reasoning: {problem}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

### 2.3 Transport Options

| Transport | Use Case | Configuration |
|-----------|----------|---------------|
| **stdio** | Local development, Claude Desktop | `mcp.run(transport="stdio")` |
| **Streamable HTTP** | Production, multi-client | `mcp.run(transport="streamable-http", port=8000)` |

### 2.4 Claude Code Integration

```bash
# Add KGoT as local MCP server
claude mcp add kgot-local -- python -m kgot.mcp.server

# Add KGoT as remote MCP server
claude mcp add --transport http kgot-remote http://localhost:8000/mcp
```

---

## 3. MCP Client Integration (Consuming External Tools)

### 3.1 Architecture Pattern

```python
# Proposed: kgot/mcp/client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.tools import BaseTool

class MCPToolWrapper(BaseTool):
    """Bridges MCP tools to LangChain interface"""

    def __init__(self, mcp_tool, session: ClientSession):
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description
        )
        self.mcp_tool = mcp_tool
        self.session = session

    async def _arun(self, **kwargs):
        result = await self.session.call_tool(self.mcp_tool.name, kwargs)
        return result.content

class MCPToolManager:
    """Manages connections to external MCP servers"""

    async def connect_to_server(self, server_name: str, config: dict):
        if config["transport"] == "stdio":
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env")
            )
            async with stdio_client(server_params) as (read, write):
                session = ClientSession(read, write)
                await session.initialize()
                tools = await session.list_tools()
                return [MCPToolWrapper(tool, session) for tool in tools.tools]
```

### 3.2 Configuration Schema

```json
// Proposed: kgot/config_mcp.json
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    },
    "firecrawl": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "firecrawl-mcp-server"],
      "env": {"FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"}
    },
    "deepseek-ocr": {
      "transport": "http",
      "url": "https://mcp-tools-deepseek-ocr-experimental.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

### 3.3 ToolManager Integration

```python
# Extend: kgot/tools/tools_v2_3/tool_manager.py
class ToolManager(ToolManagerInterface):
    def __init__(self, ..., mcp_servers_config: str = None):
        super().__init__(...)
        # ... existing init ...

        if mcp_servers_config:
            self._mcp_tools = self._load_mcp_tools(mcp_servers_config)
            self.tools.extend(self._mcp_tools)
```

---

## 4. Neo4j Vector Store Integration

### 4.1 Vector Index Architecture

```cypher
-- Create vector index on entity embeddings
CREATE VECTOR INDEX `entity-embeddings`
FOR (e:Entity) ON (e.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}

-- Hybrid search: Vector + Graph traversal
CALL db.index.vector.queryNodes('entity-embeddings', 10, $queryVector)
YIELD node, score
MATCH (node)-[r]->(related)
WHERE score > 0.5
RETURN node, r, related, score
ORDER BY score DESC
```

### 4.2 Integration with neo4j-graphrag-python

```python
# Proposed: Add to pyproject.toml
# "neo4j-graphrag[openai]>=1.14.0"

from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

class VectorEnhancedController:
    def __init__(self, driver, embedder, llm):
        self.retriever = VectorRetriever(
            driver=driver,
            index_name="entity-embeddings",
            embedder=embedder
        )
        self.rag = GraphRAG(retriever=self.retriever, llm=llm)

    def hybrid_retrieve(self, query: str, top_k: int = 5):
        # 1. Vector similarity for semantic matching
        vector_results = self.retriever.search(query_text=query, top_k=top_k)

        # 2. Graph traversal for relationship context
        cypher = """
        MATCH (n)-[r]->(m)
        WHERE n.id IN $node_ids
        RETURN n, r, m
        """
        graph_context = self.graph.get_query(cypher, node_ids=[...])

        return self._merge_results(vector_results, graph_context)
```

### 4.3 LangChain Neo4j Integration

```python
from langchain_neo4j import Neo4jVector

# Hybrid search: Vector + Keyword
db = Neo4jVector.from_documents(
    docs,
    embeddings,
    url="bolt://localhost:7687",
    search_type="hybrid",  # Key feature
    keyword_index_name="keyword_index"
)

# Similarity search with scoring
docs_with_score = db.similarity_search_with_score("query", k=5)
```

### 4.4 Benefits of Vector Enhancement

| Capability | Current KGoT | With Vector |
|------------|--------------|-------------|
| Entity matching | Exact Cypher match | Semantic similarity |
| Retrieval | Graph traversal only | Hybrid: vector seed + graph expand |
| Query flexibility | Requires precise Cypher | Natural language similarity |
| Performance | O(N) for full scan | O(log N) with HNSW index |

---

## 5. Additional Capabilities

### 5.1 GraphRAG Integration

The neo4j-graphrag-python library provides:

| Feature | Description | KGoT Application |
|---------|-------------|------------------|
| `SimpleKGPipeline` | Auto-build KG from documents | Task preprocessing |
| `VectorRetriever` | Semantic entity retrieval | Enhanced SOLVE pathway |
| `GraphRAG` | End-to-end RAG over graph | Answer synthesis |

### 5.2 Tool Dependency Management

**Research Finding:** Papers on "Graph RAG-Tool Fusion" suggest:
- Store tool metadata in knowledge graph
- Use graph traversal to discover tool dependencies
- Enable automatic tool chaining based on input/output types

```cypher
-- Tool dependency graph
(:Tool {name: "ocr"})-[:PRODUCES]->(:DataType {name: "text"})
(:DataType {name: "text"})-[:CONSUMED_BY]->(:Tool {name: "summarizer"})
```

### 5.3 Smolagents Integration

HuggingFace's `smolagents` provides a lightweight agent framework with MCP support:

```python
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")
agent = CodeAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model
)
result = agent.run("Search for information about X")
```

**Integration opportunity:** Use smolagents' MCPClient to connect KGoT to HuggingFace MCP Spaces.

---

## 6. Implementation Roadmap

### Phase 1: MCP Server (Week 1-2)

| Task | Files | Priority |
|------|-------|----------|
| Add MCP dependencies | `pyproject.toml` | High |
| Create MCP server wrapper | `kgot/mcp/server.py` | High |
| Wrap existing tools | `kgot/mcp/tool_wrapper.py` | High |
| Add stdio transport | `kgot/mcp/server.py` | Medium |
| Add HTTP transport | `kgot/mcp/server.py` | Medium |

### Phase 2: MCP Client (Week 2-3)

| Task | Files | Priority |
|------|-------|----------|
| Create MCP client | `kgot/mcp/client.py` | High |
| Create MCPToolWrapper | `kgot/mcp/client.py` | High |
| Extend ToolManager | `kgot/tools/tools_v2_3/tool_manager.py` | Medium |
| Add config schema | `kgot/config_mcp.json` | Medium |

### Phase 3: Neo4j Vector (Week 3-4)

| Task | Files | Priority |
|------|-------|----------|
| Add neo4j-graphrag dependency | `pyproject.toml` | High |
| Add vector index creation | `kgot/knowledge_graph/neo4j/main.py` | High |
| Implement hybrid retrieval | `kgot/controller/neo4j/queryRetrieve/controller.py` | Medium |
| Add embedding storage | `kgot/knowledge_graph/kg_interface.py` | Medium |

### Phase 4: Tool Upgrades (Week 4-6)

| Task | Files | Priority |
|------|-------|----------|
| Create HFToolAdapter | `kgot/tools/tools_v2_3/HFToolAdapter.py` | Medium |
| Add DeepSeek OCR tool | `kgot/tools/tools_v2_3/DeepSeekOCRTool.py` | Medium |
| Add Whisper transcription | `kgot/tools/tools_v2_3/WhisperTool.py` | Low |
| Integrate HuggingFace MCP Spaces | `kgot/config_tools.json` | Low |

---

## 7. Dependencies to Add

```toml
# pyproject.toml additions
dependencies = [
  # ... existing ...
  "mcp>=1.0.0",
  "fastmcp>=2.0.0",
  "neo4j-graphrag[openai]>=1.14.0",
  "langchain-neo4j>=0.3.0",
  "smolagents>=1.0.0",
]
```

---

## 8. Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `kgot/mcp/__init__.py` | MCP module |
| `kgot/mcp/server.py` | MCP server implementation |
| `kgot/mcp/client.py` | MCP client for external tools |
| `kgot/mcp/tool_wrapper.py` | LangChain-MCP adapters |
| `kgot/tools/tools_v2_3/HFToolAdapter.py` | HuggingFace tool base |
| `kgot/config_mcp.json` | MCP server configurations |

### Modified Files

| File | Changes |
|------|---------|
| `kgot/tools/tool_manager_interface.py` | Add MCP tool support |
| `kgot/tools/tools_v2_3/tool_manager.py` | Initialize MCP client |
| `kgot/knowledge_graph/neo4j/main.py` | Vector index methods |
| `kgot/knowledge_graph/kg_interface.py` | Vector operations |
| `kgot/controller/neo4j/queryRetrieve/controller.py` | Hybrid retrieval |
| `pyproject.toml` | New dependencies |

---

## 9. Risk Assessment

| Risk | Mitigation |
|------|------------|
| MCP protocol changes | Use official SDKs (fastmcp, mcp-agent) |
| HuggingFace API limits | Add caching, fallback to local models |
| Neo4j version compatibility | Test with Neo4j 5.11+ |
| Docker dependency for Python tool | Keep existing Docker setup as fallback |

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tool discovery time | < 100ms | MCP tools/list latency |
| Vector search latency | < 50ms | Neo4j vector query time |
| Hybrid retrieval accuracy | +20% vs pure Cypher | GAIA benchmark scores |
| External tool integration | 5+ MCP servers | Configured servers count |

---

## 11. LangChain Integration Opportunities

### 11.1 GraphCypherQAChain
**Location**: `langchain_neo4j.GraphCypherQAChain`

Simplifies Cypher generation in KGoT's Neo4j queryRetrieve controller:
```python
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

chain = GraphCypherQAChain.from_llm(
    cypher_llm=planning_llm,  # KGoT's planning LLM
    qa_llm=execution_llm,     # KGoT's execution LLM
    graph=neo4j_graph,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)
```

### 11.2 ToolStrategy for Structured Output
**Location**: `langchain.agents.structured_output.ToolStrategy`

Enforces structured output from KG reasoning:
```python
from langchain.agents.structured_output import ToolStrategy

class KGQueryResult(BaseModel):
    entities: list[str]
    relationships: list[str]
    confidence: float

agent = create_agent(
    model=execution_llm,
    tools=kgot_tools,
    response_format=ToolStrategy(KGQueryResult, handle_errors=True)
)
```

### 11.3 Tool Caching
**Location**: `langchain_anthropic` via `cache_control`

Reduces token costs for repeated tool usage:
```python
@tool(
    description="Query the knowledge graph for entity relationships",
    extras={"cache_control": {"type": "ephemeral"}}
)
def query_knowledge_graph(entity: str, relationship_type: str) -> dict:
    ...
```

### 11.4 LangGraph ToolNode
**Location**: `langgraph.prebuilt.ToolNode`

Provides stateful agent execution with tool calling loop:
```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(KGoTState)
workflow.add_node("planning", planning_llm_node)
workflow.add_node("execution", execution_llm_node)
workflow.add_node("tools", ToolNode(kgot_tools))
workflow.add_conditional_edges("execution", tools_condition)
```

---

## 12. LlamaIndex Integration Opportunities

### 12.1 PropertyGraphIndex
**Location**: `llama_index.core.indices.property_graph`

Replaces manual Cypher generation with high-level abstractions:
```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex

graph_store = Neo4jPropertyGraphStore(
    username="neo4j", password="password", url="bolt://localhost:7687"
)
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_kg_nodes=True,
)
```

### 12.2 GraphRAG with Community Detection
LlamaIndex includes hierarchical Leiden algorithm for graph partitioning:
```python
class GraphRAGStore(Neo4jPropertyGraphStore):
    def build_communities(self):
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self._summarize_communities(community_info)
```

### 12.3 ObjectIndex for Tool Retrieval
Semantic tool retrieval when dealing with many tools:
```python
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
agent = FunctionAgent(
    tool_retriever=obj_index.as_retriever(similarity_top_k=2),
    llm=OpenAI(model="gpt-4o"),
)
```

### 12.4 QueryEngineTool Pattern
Wrap each KG backend as a tool:
```python
from llama_index.core.tools import QueryEngineTool

neo4j_tool = QueryEngineTool.from_defaults(
    query_engine=neo4j_engine,
    name="neo4j_graph",
    description="Query the Neo4j knowledge graph for entity relationships"
)

networkx_tool = QueryEngineTool.from_defaults(
    query_engine=networkx_engine,
    name="networkx_graph",
    description="Query the NetworkX in-memory graph for path traversal"
)
```

---

## 13. Priority Matrix (Updated)

| Priority | Enhancement | Source | Impact | Effort |
|----------|-------------|--------|--------|--------|
| **1** | MCP Server | MCP research | High | Low |
| **2** | MCP Client | MCP research | High | Low |
| **3** | Neo4j Vector Index | Neo4j research | High | Medium |
| **4** | GraphCypherQAChain | LangChain research | High | Low |
| **5** | HuggingFace OCR Tool | HF research | Medium | Low |
| **6** | PropertyGraphIndex | LlamaIndex research | Medium | Medium |
| **7** | LangGraph ToolNode | LangChain research | Medium | Medium |
| **8** | GraphRAG Community Detection | LlamaIndex research | Medium | High |
| **9** | Tool Caching | LangChain research | Low | Low |
| **10** | ObjectIndex for Tools | LlamaIndex research | Low | Medium |

---

## 14. Sources and References

### HuggingFace
- MCP Spaces: https://huggingface.co/spaces?filter=mcp-server
- Smolagents: https://huggingface.co/docs/smolagents
- Inference API: https://huggingface.co/docs/inference-api

### MCP Protocol
- Specification: https://modelcontextprotocol.io/
- FastMCP SDK: https://github.com/punkpeye/fastmcp
- MCP Agent: https://github.com/lastmile-ai/mcp-agent

### Neo4j
- Vector Indexes: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
- GraphRAG Python: https://github.com/neo4j/neo4j-graphrag-python
- LangChain Neo4j: https://github.com/langchain-ai/langchain-neo4j

### LangChain
- GraphCypherQAChain: `langchain_neo4j`
- ToolStrategy: `langchain.agents.structured_output`
- LangGraph: https://github.com/langchain-ai/langgraph

### LlamaIndex
- PropertyGraphIndex: `llama_index.core.indices.property_graph`
- Neo4j Graph Store: `llama_index.graph_stores.neo4j`
- Documentation: https://developers.llamaindex.ai/python/

### Research Papers
- Graph RAG-Tool Fusion (arXiv:2503.01710)
- Ocean-OCR: First MLLM to outperform professional OCR (Jan 2025)
- Qianfan-OCR: SOTA on OmniDocBench (Mar 2026)

---

## 15. Multi-LLM Routing Architecture

### 15.1 Current KGoT LLM Architecture

KGoT already uses a **dual-LLM architecture** in `ControllerInterface`:

```python
# Current: kgot/controller/controller_interface.py
self.llm_planning = get_llm(llm_planning_model, llm_planning_temperature)
self.llm_execution = get_llm(llm_execution_model, llm_execution_temperature)
self.llm_math_executor = get_llm(llm_execution_model, llm_execution_temperature)
```

This provides a foundation for intelligent model routing without major refactoring.

### 15.2 LangGraph Dynamic Model Selection Pattern

LangGraph supports callable model selection via `Runtime[ModelContext]`:

```python
from langgraph.runtime import Runtime

class ModelContext(BaseModel):
    model_name: str = "gpt-4o-mini"
    complexity: str = "low"  # low, medium, high

def select_model(state: AgentState, runtime: Runtime[ModelContext]) -> BaseChatModel:
    """Dynamic model selection based on task complexity"""
    context = runtime.context

    if context.complexity == "high":
        return gpt4_model.bind_tools(tools)
    elif context.complexity == "medium":
        return sonnet_model.bind_tools(tools)
    else:
        return mini_model.bind_tools(tools)

# Usage in StateGraph
workflow.add_node("reasoning", select_model)
```

### 15.3 Proposed KGoT Routing Enhancement

**Option A: Lightweight Router (Minimal Changes)**

Add a `ModelRouter` class to `controller_interface.py`:

```python
class ModelRouter:
    """Routes to appropriate LLM based on task type"""

    ROUTING_RULES = {
        "cypher_generation": "gpt-4o-mini",      # Structured output
        "tool_selection": "gpt-4o",              # Complex reasoning
        "solution_parsing": "gpt-4o-mini",       # Simple extraction
        "math_computation": "gpt-4o",            # Numerical reasoning
    }

    def get_model(self, task_type: str) -> BaseChatModel:
        model_name = self.ROUTING_RULES.get(task_type, "gpt-4o-mini")
        return self.models[model_name]
```

**Option B: LangGraph Integration (Moderate Changes)**

Refactor KGoT to use LangGraph's `StateGraph` with conditional routing:

```python
from langgraph.graph import StateGraph

class KGoTState(TypedDict):
    problem: str
    kg_state: str
    tool_results: list
    complexity: str  # Determined by planning LLM

def route_by_complexity(state: KGoTState) -> str:
    """Conditional edge for model selection"""
    if state["complexity"] == "high":
        return "use_gpt4"
    return "use_mini"

workflow = StateGraph(KGoTState)
workflow.add_conditional_edges("planning", route_by_complexity)
```

### 15.4 Task-Based Model Selection Strategy

| Task Type | Recommended Model | Rationale |
|-----------|-------------------|-----------|
| Cypher query generation | GPT-4o-mini | Structured output, lower cost |
| Tool call decisions | GPT-4o | Complex reasoning, multi-step |
| Math computations | GPT-4o / Claude Haiku | Numerical accuracy |
| Simple text extraction | GPT-4o-mini | Cost-efficient |
| Knowledge graph retrieval | GPT-4o-mini | Pattern matching |
| Solution synthesis | GPT-4o | Complex synthesis |

### 15.5 Implementation Files

| File | Changes |
|------|---------|
| `kgot/controller/controller_interface.py` | Add `ModelRouter` class |
| `kgot/controller/neo4j/queryRetrieve/llm_invocation_handle.py` | Use router for model selection |
| `kgot/utils/llm_utils.py` | Support multiple model instances |
| `kgot/config_llms.json` | Add routing configuration |

---

## 16. Web Scraping and Research Tools

### 16.1 Current Capabilities

KGoT has basic web search via:
- `SearchTool` (SurferTool) - SerpAPI-based search
- `PollinationsSearchTool` - AI-powered search

**Gap:** No dedicated web scraping/crawling capability for deep research.

### 16.2 Recommended Tools

#### Firecrawl (97k+ GitHub stars) - **Primary Recommendation**

**Why:** Production-grade, LLM-ready output, MCP server available.

```bash
# MCP Server Installation
npm install -g firecrawl-mcp-server
# Or via npx
npx firecrawl-mcp-server
```

```python
# Integration as LangChain Tool
from firecrawl import FirecrawlApp

class FirecrawlTool(BaseTool):
    name = "firecrawl_scrape"
    description = "Scrape and convert web pages to LLM-ready markdown"

    def _run(self, url: str, formats: list = ["markdown"]):
        app = FirecrawlApp(api_key=self.api_key)
        return app.scrape_url(url, params={"formats": formats})
```

**Key Features:**
- LLM-ready markdown output
- Automatic JavaScript rendering
- Structured data extraction via JSON schema
- Batch crawling with rate limiting
- MCP server for Claude Code integration

#### Crawl4AI (62k+ GitHub stars) - **Open Source Alternative**

```python
# kgot/tools/tools_v2_3/Crawl4AITool.py
from crawl4ai import AsyncWebCrawler

class Crawl4AITool(BaseTool):
    name = "web_crawler"
    description = "Crawl websites with LLM-friendly extraction"

    async def _arun(self, url: str, extract_links: bool = True):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return {
                "markdown": result.markdown,
                "links": result.links if extract_links else None,
                "metadata": result.metadata
            }
```

**Key Features:**
- Fully open source (no API key required)
- Built-in LLM extraction strategies
- Proxy rotation support
- Sitemap and RSS discovery

#### Pydoll (6.6k+ GitHub stars) - **Browser Automation**

For sites requiring JavaScript interaction:

```python
# kgot/tools/tools_v2_3/PydollTool.py
from pydoll.browser import Browser

class PydollTool(BaseTool):
    name = "browser_automation"
    description = "Automate browser interactions for dynamic content"

    async def _arun(self, url: str, actions: list):
        async with Browser() as browser:
            page = await browser.get(url)
            for action in actions:
                if action["type"] == "click":
                    await page.click(action["selector"])
                elif action["type"] == "type":
                    await page.type(action["selector"], action["text"])
            return await page.content()
```

**Key Features:**
- No WebDriver dependency (native Chrome DevTools Protocol)
- Built-in anti-detection measures
- Faster than Selenium/Playwright for scraping

### 16.3 Tool Comparison Matrix

| Feature | Firecrawl | Crawl4AI | Pydoll |
|---------|-----------|----------|--------|
| Setup complexity | Low (SaaS) | Low (pip) | Medium |
| JS rendering | Yes | Yes | Yes |
| LLM-ready output | Yes | Yes | No |
| Anti-detection | Yes | Partial | Yes |
| MCP support | Yes | No | No |
| Cost | Freemium | Free | Free |
| Best for | Production scraping | Self-hosted | Interactive sites |

### 16.4 Recommended Integration

1. **Primary:** Add Firecrawl MCP server to `config_mcp.json`
2. **Fallback:** Implement Crawl4AI as native LangChain tool
3. **Special cases:** Use Pydoll for authenticated/bot-protected sites

```json
// config_mcp.json addition
{
  "mcpServers": {
    "firecrawl": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "firecrawl-mcp-server"],
      "env": {"FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"}
    }
  }
}
```

---

## 17. Advanced Python Execution Enhancement

### 17.1 Current State

**Container:** `containers/python/files/python_executor.py`
**Libraries:** flask, waitress, langchain_experimental, requests (only 4 packages)
**Features:**
- 240-second timeout
- Auto pip install for `required_modules`
- Auto-fix capability (up to 3 retries)
- ThreadPoolExecutor for concurrent execution

**Limitations:**
- No data science libraries (pandas, numpy, matplotlib)
- No visualization output
- No persistent state between calls
- No file system access

### 17.2 Enhancement Proposals

#### A. Expanded Library Stack

Add to `containers/python/files/requirements.txt`:

```txt
# Data Science
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# ML/Lightweight
scikit-learn>=1.4.0
lightgbm>=4.3.0

# Web/Data
requests>=2.31.0
beautifulsoup4>=4.12.0
httpx>=0.26.0

# Utilities
pydantic>=2.5.0
python-dateutil>=2.8.0
```

#### B. Visualization Output Support

Modify `python_executor.py` to handle plot outputs:

```python
import base64
import io
import matplotlib.pyplot as plt

class PythonExecutor:
    def execute_with_viz(self, code: str) -> dict:
        """Execute code and capture visualizations"""
        result = super().execute(code)

        # Capture any matplotlib figures
        figures = []
        for i, fig in enumerate(plt.get_fignums()):
            buf = io.BytesIO()
            plt.figure(fig).savefig(buf, format='png')
            figures.append({
                "figure_index": i,
                "base64": base64.b64encode(buf.getvalue()).decode()
            })
        plt.close('all')

        return {
            **result,
            "figures": figures
        }
```

#### C. Persistent Session State

Enable stateful execution sessions:

```python
class SessionManager:
    """Manages persistent Python sessions"""

    def __init__(self):
        self.sessions = {}  # session_id -> SessionState

    def create_session(self, session_id: str) -> dict:
        """Create a new isolated session"""
        self.sessions[session_id] = {
            "globals": {},
            "imports": set(),
            "created_at": time.time()
        }
        return {"session_id": session_id, "status": "created"}

    def execute_in_session(self, session_id: str, code: str) -> dict:
        """Execute code in session context (variables persist)"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        exec(code, session["globals"])
        return {"status": "success", "variables": list(session["globals"].keys())}
```

#### D. HuggingFace MCP Alternative

Use `vmohan-sn/PythonCodeExec` MCP Space for cloud execution:

```python
# MCP-native Python execution
class HFPythonTool(BaseTool):
    name = "hf_python_executor"
    description = "Execute Python code via HuggingFace MCP Space"

    def _run(self, code: str):
        # Call vmohan-sn/PythonCodeExec via MCP
        ...
```

### 17.3 Security Considerations

| Enhancement | Security Risk | Mitigation |
|-------------|---------------|------------|
| Expanded libraries | Increased attack surface | Pin versions, audit dependencies |
| Visualization output | Memory exhaustion | Limit figure count/size |
| Session persistence | State pollution | Session isolation, timeout cleanup |
| File system access | Data exfiltration | Sandbox with restricted paths |

### 17.4 Implementation Priority

| Priority | Enhancement | Effort | Impact |
|----------|-------------|--------|--------|
| **1** | Add pandas/numpy/matplotlib | Low | High |
| **2** | Visualization capture | Medium | High |
| **3** | HF MCP Python executor | Low | Medium |
| **4** | Session persistence | Medium | Medium |

---

## 18. LangGraph Integration Patterns

### 18.1 Why LangGraph?

LangGraph provides:
- **Stateful agent workflows** - Persist state across steps
- **Conditional routing** - Route based on LLM decisions
- **Built-in tool loop** - `ToolNode` handles tool calling automatically
- **Checkpointing** - Resume interrupted workflows
- **Human-in-the-loop** - Approval gates for sensitive actions

### 18.2 KGoT as LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

class KGoTState(TypedDict):
    problem: str
    kg_state: str
    tool_calls: list
    solutions: list[str]
    iteration: int
    max_iterations: int

def build_kgot_workflow(tools: list, planning_llm, execution_llm):
    workflow = StateGraph(KGoTState)

    # Nodes
    workflow.add_node("decide_next_step", planning_llm_node)
    workflow.add_node("execute_tools", ToolNode(tools))
    workflow.add_node("insert_to_kg", insert_node)
    workflow.add_node("retrieve_from_kg", retrieve_node)
    workflow.add_node("synthesize_solution", synthesis_node)

    # Entry point
    workflow.set_entry_point("decide_next_step")

    # Conditional routing
    def route_decision(state: KGoTState) -> str:
        if state["iteration"] >= state["max_iterations"]:
            return "retrieve"
        if state["decision"] == "INSERT":
            return "insert"
        return "retrieve"

    workflow.add_conditional_edges(
        "decide_next_step",
        route_decision,
        {"insert": "execute_tools", "retrieve": "retrieve_from_kg"}
    )

    workflow.add_edge("execute_tools", "insert_to_kg")
    workflow.add_edge("insert_to_kg", "decide_next_step")
    workflow.add_edge("retrieve_from_kg", "synthesize_solution")
    workflow.add_edge("synthesize_solution", END)

    return workflow.compile()
```

### 18.3 Deer-Flow Architecture Reference

**Deer-Flow** (40k+ GitHub stars) is a multi-agent framework from ByteDance built on LangGraph:

```python
# deer-flow pattern: Hierarchical agents
from langgraph.graph import StateGraph

class DeerFlowPattern:
    """
    Key patterns from deer-flow:
    1. Orchestrator agent delegates to specialist agents
    2. Each specialist has its own StateGraph
    3. Shared context via state passing
    """

    def build_research_agent(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("search", self.search_node)
        workflow.add_node("extract", self.extract_node)
        workflow.add_node("synthesize", self.synthesize_node)
        return workflow.compile()

    def build_analysis_agent(self):
        workflow = StateGraph(AnalysisState)
        workflow.add_node("parse", self.parse_node)
        workflow.add_node("compute", self.compute_node)
        return workflow.compile()
```

### 18.4 Integration Opportunities

| KGoT Component | LangGraph Equivalent | Benefit |
|----------------|---------------------|---------|
| `_iterative_next_step_logic` | StateGraph with loop | Checkpointing, observability |
| `_invoke_tools_after_llm_response` | `ToolNode` | Automatic tool loop |
| `define_next_step` | Conditional edges | Explicit routing logic |
| Dual LLM architecture | Callable model selection | Dynamic model routing |
| Tool call cache | State persistence | Native caching |

### 18.5 Migration Strategy

**Phase 1: Parallel Implementation**
- Create `kgot/controller/langgraph/` directory
- Implement KGoT as LangGraph workflow alongside existing controllers
- Run benchmarks to compare performance

**Phase 2: Gradual Adoption**
- Replace `_invoke_tools_after_llm_response` with `ToolNode`
- Replace `_iterative_next_step_logic` with StateGraph loop
- Add checkpointing for long-running tasks

**Phase 3: Full Integration**
- Conditional model selection based on task complexity
- Human-in-the-loop approval for high-impact KG writes
- Resume capability for interrupted sessions

### 18.6 Implementation Files

| New File | Purpose |
|----------|---------|
| `kgot/controller/langgraph/__init__.py` | LangGraph controller module |
| `kgot/controller/langgraph/controller.py` | LangGraph-based controller |
| `kgot/controller/langgraph/state.py` | State definitions |
| `kgot/controller/langgraph/nodes.py` | Node implementations |
| `kgot/controller/langgraph/routing.py` | Routing logic |

---

## 19. Conclusion

This proposal outlines a comprehensive enhancement strategy for KGoT that:

1. **Enables MCP protocol support** - Both as server (exposing tools) and client (consuming tools)
2. **Adds vector search capabilities** - Hybrid retrieval combining semantic similarity with graph traversal
3. **Integrates HuggingFace ecosystem** - Access to MCP Spaces, Inference API, and smolagents
4. **Implements multi-LLM routing** - Task-based model selection for cost/quality optimization
5. **Adds web scraping tools** - Firecrawl MCP, Crawl4AI, and Pydoll for research capabilities
6. **Enhances Python execution** - Data science libraries, visualization support, session persistence
7. **Enables LangGraph workflows** - Stateful agents, conditional routing, checkpointing
8. **Maintains backward compatibility** - All changes are additive to existing architecture

**Recommended Next Steps:**
1. Begin with Phase 1 (MCP Server) for immediate usability from Claude Code
2. Add web scraping tools (Firecrawl MCP) for enhanced research capabilities
3. Expand Python execution libraries for data science tasks
4. Implement multi-LLM routing for cost optimization