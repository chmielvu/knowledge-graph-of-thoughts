# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge Graph of Thoughts (KGoT) is an AI assistant architecture that integrates LLM reasoning with dynamically constructed knowledge graphs. It extracts and structures task-relevant knowledge into a dynamic KG representation, iteratively enhanced through external tools.

## Core Architecture

The system has three main components:

### 1. Controller (`kgot/controller/`)
The central orchestrator managing the interaction between the knowledge graph and integrated tools. Uses a dual-LLM architecture:
- **LLM Graph Executor**: Decision making and KG-based reasoning (SOLVE/ENHANCE pathways)
- **LLM Tool Executor**: Decides which tools to use and handles interactions

Key parameters configurable in controllers:
- `num_next_steps_decision`: Number of times to prompt LLM on SOLVE vs ENHANCE
- `max_retrieve_query_retry`: Max retries for SOLVE queries
- `max_cypher_fixing_retry`: Max retries for fixing Cypher queries
- `max_tool_retries`: Max retries when tool invocation fails

### 2. Graph Store (`kgot/knowledge_graph/`)
Manages the dynamically evolving KG. Supports three backends:
- **Neo4j**: Uses Cypher queries; good for retrieving specific subgraphs/patterns
- **FalkorDB**: Redis-based graph database with semantic search capabilities
- **NetworkX**: In-memory lightweight alternative using Python `exec()` for queries; good for path traversals with computational steps

All backends implement `KnowledgeGraphInterface` in `kg_interface.py`.

### 3. Integrated Tools (`kgot/tools/`)
Enables multi-modal reasoning capabilities:
- `RunPythonCodeTool`: Executes Python code in sandboxed Docker environment
- `WebBrowserTool`: Web searches via browser automation
- `TextInspectorTool`: Examines text files, PDFs, spreadsheets, etc.
- `ImageQuestionTool`: Analyzes images with vision models
- `ExtractZipTool`: Handles compressed files
- `LangchainLLMTool`: Additional LLM for extended knowledge
- `GraphVizTool`: Graph visualization
- `PollinationsSearchTool`: AI-powered search tool
- `FalkorDBSearchTool`: Semantic search on knowledge graph

Tools must adhere to LangChain's `BaseTool` interface.

## Commands

### Setup
```bash
pip install -e .
playwright install
```

### Running KGoT
```bash
# Single problem (graph backend is fixed to FalkorDB)
kgot single -p "What is a knowledge graph?"

# With files and a specific controller
kgot --controller_choice directRetrieve single -p "Summarize these files" --files path/to/file1 --files path/to/file2

# With custom LLM models
kgot --llm-plan gpt-4o-mini --llm-exec gpt-4o-mini --iterations 7 single -p "Your question"
```
### Running Benchmarks
```bash
# GAIA benchmark
./run_multiple_gaia.sh --log_base_folder logs/test --controller_choice queryRetrieve --backend_choice networkX --max_iterations 5

# SimpleQA benchmark
./run_multiple_simpleqa.sh
```

### Docker Containers
```bash
cd containers/
docker compose up  # Starts Neo4j and Python executor
cd containers/kgot/
docker compose up  # Starts KGoT container
```

## Configuration

Required config files (copy from templates):
- `kgot/config_llms.json` - LLM API keys and model configurations
- `kgot/config_tools.json` - Tool API keys (Pollinations)

Environment variables (`.env` file):
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `FALKORDB_HOST`, `FALKORDB_PORT`, `FALKORDB_GRAPH_NAME`
- `PYTHON_EXECUTOR_URI` (default: `http://localhost:16000/run`)

## Controller Types

Two retrieval strategies available:
- **directRetrieve**: Embeds the entire KG in LLM context for solution
- **queryRetrieve**: Uses LLM-generated queries (Cypher/Python) to extract specific information

File structure for controllers:
```
kgot/controller/{backend}/{retrieval_type}/controller.py
kgot/controller/{backend}/{retrieval_type}/llm_invocation_handle.py
```

## Prompts

Prompts are organized by backend and retrieval type:
```
kgot/prompts/{backend}/{retrieval_type}/prompts.py
kgot/prompts/tools/tools_v2_3/
```

## Adding New Tools

1. Create tool class inheriting from LangChain's `BaseTool`
2. Initialize in `kgot/tools/tools_v2_3/tool_manager.py`
3. Append to `self.tools` list in `ToolManager.__init__`

## Adding New Graph Backend

1. Create new directory in `kgot/knowledge_graph/`
2. Implement `KnowledgeGraphInterface` abstract class
3. Create corresponding controller in `kgot/controller/`
4. Create prompts in `kgot/prompts/`

## Testing Python Docker Service

The `ToolManager` automatically tests the Python Docker service on initialization. Ensure the container is running before starting KGoT with the Python tool.

## Python Version

Requires Python `>=3.10,<3.13`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **knowledge-graph-of-thoughts** (2247 symbols, 4310 relationships, 190 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/knowledge-graph-of-thoughts/context` | Codebase overview, check index freshness |
| `gitnexus://repo/knowledge-graph-of-thoughts/clusters` | All functional areas |
| `gitnexus://repo/knowledge-graph-of-thoughts/processes` | All execution flows |
| `gitnexus://repo/knowledge-graph-of-thoughts/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
