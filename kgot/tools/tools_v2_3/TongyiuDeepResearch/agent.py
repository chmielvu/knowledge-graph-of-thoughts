"""
Deep Research Agent

LangGraph-based research agent with:
- Multi-step research workflow
- Conditional edges for iteration
- Memory checkpointing
- Tool integration
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Literal, Optional, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from .configuration import DeepResearchConfig
from .state import ResearchState, Source
from .tools import ResilientSearchTool, WikipediaTool

logger = logging.getLogger(__name__)


# =============================================================================
# Research Agent Workflow
# =============================================================================

# System prompt for the research agent
RESEARCH_SYSTEM_PROMPT = """You are a deep research agent. Your role is to conduct thorough, multi-source investigations.

# Research Methodology

1. **Discovery Phase**: Generate targeted search queries to find relevant sources
2. **Deep Dive Phase**: Analyze promising sources for detailed information
3. **Synthesis Phase**: Combine findings into coherent insights
4. **Reflection Phase**: Identify gaps and determine if more research is needed

# Quality Standards

- Always cite sources using [source: URL] format
- Distinguish between verified facts and claims needing verification
- Note contradictions between sources
- Assess source credibility and recency
- Identify knowledge gaps when present

# Instructions

When given a research topic:
1. Generate 2-3 targeted search queries
2. Analyze the search results
3. If more information is needed, identify specific gaps
4. Synthesize findings into a clear, comprehensive response

When research is complete, provide your answer enclosed in <answer></answer> tags.

Current date: {current_date}"""


def create_research_agent(
    config: Optional[DeepResearchConfig] = None,
    tools: Optional[Dict[str, BaseTool]] = None
) -> StateGraph:
    """
    Create the research agent workflow using LangGraph.

    The workflow has 4 nodes:
    1. generate_query - Create optimized search queries
    2. web_research - Execute searches and gather sources
    3. summarize - Synthesize findings
    4. reflect - Identify gaps and determine next steps

    Returns a compiled StateGraph ready for execution.
    """
    config = config or DeepResearchConfig.from_env()

    # Initialize tools if not provided
    if not tools:
        tools = {
            "web_search": ResilientSearchTool(config=config),
            "wikipedia_search": WikipediaTool(),
        }

    # Initialize orchestrator LLM (Tongyi via nanoGPT)
    orchestrator = ChatOpenAI(
        model=config.tongyi_model,
        api_key=config.nanogpt_api_key,
        base_url=config.nanogpt_base_url,
        temperature=0.1,
        max_retries=3
    )

    # Create the workflow
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("generate_query", lambda state: generate_query(state, orchestrator))
    workflow.add_node("web_research", lambda state: web_research(state, tools))
    workflow.add_node("summarize", lambda state: summarize(state, orchestrator))
    workflow.add_node("reflect", lambda state: reflect(state, orchestrator))

    # Define edges
    workflow.set_entry_point("generate_query")
    workflow.add_edge("generate_query", "web_research")
    workflow.add_edge("web_research", "summarize")
    workflow.add_edge("summarize", "reflect")

    # Conditional edge: continue research or end
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "continue": "generate_query",
            "complete": END
        }
    )

    # Compile with memory
    return workflow.compile(checkpointer=MemorySaver())


# =============================================================================
# Node Functions
# =============================================================================

def generate_query(state: ResearchState, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Generate optimized search queries from the research topic.

    Uses the LLM to create targeted queries that address:
    - The main research topic
    - Any identified gaps from previous iterations
    - Focus areas if specified
    """
    logger.info(f"Generating query for: {state.research_topic}")

    # Build context from previous iterations
    context = f"Research topic: {state.research_topic}"

    if state.gaps_identified:
        context += f"\n\nIdentified gaps to address: {', '.join(state.gaps_identified)}"

    if state.research_loop_count > 0:
        context += f"\n\nPrevious queries: {[q.query for q in state.queries_generated]}"

    prompt = f"""Generate 1-3 targeted search queries to research this topic.

{context}

Return ONLY the queries, one per line, without numbering or explanation.
Focus on finding specific, authoritative sources."""

    try:
        response = llm.invoke(prompt)
        queries_text = response.content if hasattr(response, 'content') else str(response)
        queries = [q.strip() for q in queries_text.strip().split('\n') if q.strip()][:3]

        # Use the first query
        primary_query = queries[0] if queries else state.research_topic

        return {
            "search_query": primary_query,
        }
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        return {"search_query": state.research_topic}


def web_research(state: ResearchState, tools: Dict[str, BaseTool]) -> Dict[str, Any]:
    """
    Execute web research using available tools.

    Performs search and optionally visits promising results.
    Updates the state with gathered sources.
    """
    logger.info(f"Executing research for: {state.search_query}")
    config = state.config or DeepResearchConfig.from_env()

    search_tool = tools.get("web_search")
    if not search_tool:
        return {"web_research_results": "Error: Search tool not available"}

    try:
        # Execute search (sync wrapper)
        results = search_tool.invoke({
            "query": state.search_query,
            "max_results": config.max_search_results,
            "include_wikipedia": config.include_wikipedia
        })

        # Track sources
        new_sources = extract_sources_from_results(results)
        for source in new_sources:
            state.add_source(source)

        return {
            "web_research_results": results,
            "research_loop_count": state.research_loop_count + 1
        }
    except Exception as e:
        logger.error(f"Web research failed: {e}")
        return {
            "web_research_results": f"Search error: {str(e)}\n\nTry different search terms.",
            "research_loop_count": state.research_loop_count + 1
        }


def summarize(state: ResearchState, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Synthesize gathered sources into a coherent summary.

    Uses the orchestrator LLM to combine findings and
    identify key insights.
    """
    logger.info("Summarizing research findings")

    sources_text = "\n\n---\n\n".join([
        f"Source: {s.url}\n{s.content[:1000] if s.content else s.snippet}"
        for s in state.sources_gathered[-5:]  # Last 5 sources
    ])

    prompt = f"""Based on the following research, provide a concise summary.

Research topic: {state.research_topic}

Sources:
{sources_text[:8000]}

Provide:
1. Key findings (bullet points)
2. Overall assessment
3. Any gaps or areas needing more research

Be specific and cite sources using [source: URL] format."""

    try:
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)

        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return {"summary": f"Summarization error: {str(e)}"}


def reflect(state: ResearchState, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Reflect on the current research and determine next steps.

    Identifies gaps and decides whether more research is needed.
    """
    logger.info(f"Reflecting on research (loop {state.research_loop_count})")
    config = state.config or DeepResearchConfig.from_env()

    # Check termination conditions
    if state.research_loop_count >= config.max_research_loops:
        return {"is_complete": True, "gaps_identified": []}

    prompt = f"""Analyze this research and determine if it's sufficient.

Research topic: {state.research_topic}
Research iterations: {state.research_loop_count}
Sources found: {len(state.sources_gathered)}

Summary:
{state.summary[:2000]}

1. Is the research sufficient? (yes/no)
2. If no, what specific gaps remain? (list them)

Be strict - only continue if critical information is missing."""

    try:
        response = llm.invoke(prompt)
        reflection = response.content if hasattr(response, 'content') else str(response)

        # Parse response
        is_sufficient = "yes" in reflection.lower().split('\n')[0] if reflection else True

        # Extract gaps
        gaps = []
        if not is_sufficient:
            # Look for gap list
            lines = reflection.split('\n')
            for line in lines:
                if line.strip().startswith(('-', '*', '•')):
                    gaps.append(line.strip().lstrip('-*• ').strip())

        if is_sufficient or len(state.sources_gathered) >= config.max_research_loops * config.max_sources_per_query:
            return {"is_complete": True, "gaps_identified": []}

        return {"gaps_identified": gaps[:3]}

    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        # Default to completing if reflection fails
        return {"is_complete": True, "gaps_identified": []}


def should_continue_research(state: ResearchState) -> Literal["continue", "complete"]:
    """
    Determine if the research loop should continue.

    Returns:
        "continue" if more research is needed
        "complete" if research is finished
    """
    if state.is_complete:
        return "complete"
    if not state.gaps_identified:
        return "complete"
    if state.research_loop_count >= (state.config.max_research_loops if state.config else 3):
        return "complete"
    return "continue"


# =============================================================================
# Helper Functions
# =============================================================================

def extract_sources_from_results(results: str) -> list:
    """Extract Source objects from formatted search results."""
    import re
    sources = []

    # Pattern for search results
    pattern = r'\*\*(\d+)\.\s+(.+?)\*\*\s*\[([^\]]+)\]\(([^)]+)\)\s*([^\*]+?)(?=\*\*\d+|$)'
    matches = re.findall(pattern, results, re.DOTALL)

    for match in matches:
        num, title, url_text, url, snippet = match
        sources.append(Source(
            title=title.strip(),
            url=url.strip(),
            snippet=snippet.strip()[:500],
            source_type="web"
        ))

    return sources


# =============================================================================
# Main Research Function
# =============================================================================

async def run_research(
    query: str,
    depth: Literal["quick", "standard", "deep"] = "standard",
    config: Optional[DeepResearchConfig] = None
) -> ResearchState:
    """
    Execute deep research on a topic.

    Args:
        query: Research topic or question
        depth: Research depth ('quick'=1 loop, 'standard'=3 loops, 'deep'=5 loops)
        config: Optional configuration

    Returns:
        Final ResearchState with all gathered information
    """
    config = config or DeepResearchConfig.from_env()

    # Adjust max loops based on depth
    depth_loops = {"quick": 1, "standard": 3, "deep": 5}
    config.max_research_loops = depth_loops.get(depth, 3)

    # Create agent
    agent = create_research_agent(config)

    # Initialize state
    initial_state = ResearchState(
        research_topic=query,
        config=config
    )

    # Run the agent
    result = await agent.ainvoke(initial_state)

    return result