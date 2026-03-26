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
- **NetworkX**: In-memory lightweight alternative using Python `exec()` for queries; good for path traversals with computational steps
- **RDF4J**: Uses SPARQL queries; requires read and write endpoints

All backends implement `KnowledgeGraphInterface` in `kg_interface.py`.

### 3. Integrated Tools (`kgot/tools/`)
Enables multi-modal reasoning capabilities:
- `RunPythonCodeTool`: Executes Python code in sandboxed Docker environment
- `SearchTool` (Surfer Agent): Web searches via SerpAPI
- `TextInspectorTool`: Examines text files, PDFs, spreadsheets, etc.
- `ImageQuestionTool`: Analyzes images with vision models
- `ExtractZipTool`: Handles compressed files
- `LangchainLLMTool`: Additional LLM for extended knowledge
- `GraphVizTool`: Graph visualization
- `PollinationsSearchTool`: Alternative search tool

Tools must adhere to LangChain's `BaseTool` interface.

## Commands

### Setup
```bash
pip install -e .
playwright install
```

### Running KGoT
```bash
# Single problem
kgot single -p "What is a knowledge graph?"

# With files and specific backend
kgot --db_choice neo4j --controller_choice directRetrieve single -p "Summarize these files" --files path/to/file1 path/to/file2

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
- `kgot/config_tools.json` - Tool API keys (SerpAPI for Surfer Agent)

Environment variables (`.env` file):
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `PYTHON_EXECUTOR_URI` (default: `http://localhost:16000/run`)
- `RDF4J_READ_URI`, `RDF4J_WRITE_URI`

## Controller Types

Two retrieval strategies available:
- **directRetrieve**: Embeds the entire KG in LLM context for solution
- **queryRetrieve**: Uses LLM-generated queries (Cypher/Python/SPARQL) to extract specific information

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