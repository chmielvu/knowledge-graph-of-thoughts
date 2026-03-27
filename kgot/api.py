# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# FastAPI wrapper for KGoT - exposes agent as local REST API
# Run with: uvicorn kgot.api:app --reload --port 8000

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="KGoT API",
    description="Knowledge Graph of Thoughts - AI Agent REST API",
    version="1.1.0",
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for KGoT queries."""

    problem: str = Field(..., description="The problem statement to solve")
    files: list[str] = Field(default=[], description="List of file paths associated with the problem")

    # Database configuration
    db_choice: str = Field(default="falkordb", description="Database backend (neo4j, falkordb, networkX)")
    controller_choice: str = Field(default="queryRetrieve", description="Controller type")

    # LLM configuration
    llm_plan: str = Field(default="gpt-4o-mini", description="LLM model for planning")
    llm_exec: str = Field(default="gpt-4o-mini", description="LLM model for execution")
    llm_plan_temp: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_exec_temp: float = Field(default=0.0, ge=0.0, le=2.0)

    # Iteration limits
    max_iterations: int = Field(default=7, ge=1, le=20)

    # Semantic search configuration
    enable_semantic_search: bool = Field(default=True, description="Enable semantic search tools")


class QueryResponse(BaseModel):
    """Response model for KGoT queries."""

    solution: str = Field(..., description="The final solution/answer")
    iterations: int = Field(default=0, description="Number of iterations used")
    graph_state: dict[str, Any] = Field(default={}, description="Final knowledge graph state")
    tool_calls: list[dict] = Field(default=[], description="List of tool calls made")


class SearchRequest(BaseModel):
    """Request model for direct semantic search."""

    query: str = Field(..., description="Search query")
    k: int = Field(default=5, ge=1, le=20, description="Number of results")
    search_type: str = Field(default="hybrid", description="Search type: semantic, hybrid, or fulltext")


class SearchResult(BaseModel):
    """Result model for semantic search."""

    results: list[dict[str, Any]]
    query: str
    total: int


# Global controller cache
_controller_cache: dict[str, Any] = {}


def get_controller(
    db_choice: str = "falkordb",
    controller_choice: str = "queryRetrieve",
    llm_plan: str = "gpt-4o-mini",
    llm_exec: str = "gpt-4o-mini",
    enable_semantic_search: bool = True,
    **kwargs,
) -> Any:
    """Get or create a controller instance."""
    import importlib
    import os

    cache_key = f"{db_choice}_{controller_choice}_{llm_plan}_{llm_exec}"

    if cache_key not in _controller_cache:
        # Load environment
        from dotenv import load_dotenv

        load_dotenv()

        # Get environment variables
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        python_executor_uri = os.getenv("PYTHON_EXECUTOR_URI", "http://localhost:16000/run")
        mistral_api_key = os.getenv("MISTRAL_API_KEY")

        # Import controller dynamically
        controller_module = importlib.import_module(f"kgot.controller.{db_choice}.{controller_choice}")
        controller_class = getattr(controller_module, "Controller")

        # Create controller with appropriate parameters
        controller_params: dict[str, Any] = {
            "neo4j_uri": neo4j_uri,
            "neo4j_username": neo4j_user,
            "neo4j_pwd": neo4j_password,
            "python_executor_uri": python_executor_uri,
            "llm_planning_model": llm_plan,
            "llm_execution_model": llm_exec,
            "db_choice": db_choice,
            "controller_choice": controller_choice,
        }

        # Add FalkorDB-specific parameters
        if db_choice == "falkordb":
            falkordb_host = os.getenv("FALKORDB_HOST", "localhost")
            falkordb_port = int(os.getenv("FALKORDB_PORT", "6379"))
            falkordb_password = os.getenv("FALKORDB_PASSWORD")

            controller_params.update(
                {
                    "falkordb_host": falkordb_host,
                    "falkordb_port": falkordb_port,
                    "falkordb_password": falkordb_password,
                    "mistral_api_key": mistral_api_key,
                    "enable_semantic_search": enable_semantic_search,
                }
            )

        _controller_cache[cache_key] = controller_class(**controller_params)

    return _controller_cache[cache_key]


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "KGoT API", "version": "1.1.0"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a problem to KGoT for solving.

    The agent will:
    1. Analyze the problem
    2. Build/enhance a knowledge graph
    3. Use tools to gather information
    4. Return a solution
    """
    try:
        controller = get_controller(
            db_choice=request.db_choice,
            controller_choice=request.controller_choice,
            llm_plan=request.llm_plan,
            llm_exec=request.llm_exec,
            enable_semantic_search=request.enable_semantic_search,
        )

        # Run the controller
        solution, iterations = controller.run(
            problem=request.problem,
            attachments_file_path="",
            attachments_file_names=request.files,
        )

        # Get final graph state
        graph_state = controller.graph.get_current_graph_state() if hasattr(controller, "graph") else {}

        return QueryResponse(
            solution=solution,
            iterations=iterations,
            graph_state=graph_state,
            tool_calls=getattr(controller, "tool_calls_made", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResult)
async def search(request: SearchRequest) -> SearchResult:
    """
    Direct semantic search on the knowledge graph.

    Requires FalkorDB with semantic search enabled.
    """
    try:
        controller = get_controller(db_choice="falkordb", enable_semantic_search=True)
        kg = controller.graph

        if request.search_type == "semantic":
            results = kg.semantic_search(request.query, k=request.k)
        elif request.search_type == "hybrid":
            results = kg.hybrid_search(request.query, k=request.k)
        elif request.search_type == "fulltext":
            results = kg.fulltext_search_documents(request.query, k=request.k)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown search type: {request.search_type}")

        # Convert Documents to dicts
        result_dicts = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]

        return SearchResult(
            results=result_dicts,
            query=request.query,
            total=len(result_dicts),
        )
    except AttributeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Search type not available. Ensure FalkorDB with semantic search is configured. Error: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/state")
async def get_graph_state(db_choice: str = "falkordb") -> dict[str, Any]:
    """Get the current state of the knowledge graph."""
    try:
        controller = get_controller(db_choice=db_choice)
        return controller.graph.get_current_graph_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/graph/clear")
async def clear_graph(db_choice: str = "falkordb") -> dict[str, str]:
    """Clear the knowledge graph (WARNING: destructive operation)."""
    try:
        controller = get_controller(db_choice=db_choice)
        controller.graph.clear_graph()
        return {"status": "ok", "message": "Graph cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def list_tools(db_choice: str = "falkordb", controller_choice: str = "queryRetrieve") -> list[dict[str, str]]:
    """List available tools for the agent."""
    try:
        controller = get_controller(db_choice=db_choice, controller_choice=controller_choice)
        return [
            {
                "name": tool.name,
                "description": tool.description[:200] + "..." if len(tool.description) > 200 else tool.description,
            }
            for tool in controller.tools
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For running directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)