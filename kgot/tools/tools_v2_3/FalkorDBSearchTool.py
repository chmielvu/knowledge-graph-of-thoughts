# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# FalkorDB Search Tools for KGoT
#
# Provides semantic, hybrid, and text-to-cypher search capabilities

from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# LangChain FalkorDB integrations
from langchain_community.graphs import FalkorDBGraph
from langchain_community.chains.graph_qa.falkordb import FalkorDBQAChain

# LLM imports
from langchain_mistralai import ChatMistralAI


# =============================================================================
# Input Schemas
# =============================================================================

class SemanticSearchSchema(BaseModel):
    """Input schema for semantic search tool."""
    query: str = Field(
        description="Natural language query for semantic search. Be specific about what concepts or ideas you're looking for."
    )
    k: int = Field(default=5, description="Number of results to return (1-20)")


class HybridSearchSchema(BaseModel):
    """Input schema for hybrid search tool."""
    query: str = Field(
        description="Search query combining keywords and conceptual terms. Good for general searches."
    )
    k: int = Field(default=5, description="Number of results to return (1-20)")


class TextToCypherSchema(BaseModel):
    """Input schema for text-to-cypher tool."""
    question: str = Field(
        description="Natural language question about entities and relationships in the graph. Examples: 'What entities are connected to X?', 'How many nodes of type Y exist?'"
    )


# =============================================================================
# Semantic Search Tool
# =============================================================================

class SemanticSearchTool(BaseTool):
    """Tool for semantic (vector similarity) search on the knowledge graph.

    Uses Mistral embeddings to find conceptually similar content.
    Best for finding related concepts and ideas, not exact matches.
    """
    name: str = "semantic_search"
    description: str = """Search the knowledge graph using vector embeddings for semantic similarity.
Use when you need to find conceptually related content, similar ideas, or when you don't know exact keywords.
Returns the most semantically similar nodes with similarity scores.

Good examples:
- "concepts related to machine learning"
- "ideas similar to knowledge representation"
- "approaches for graph traversal"

Avoid:
- Single keywords: "AI", "graph"
- Very long queries (>50 words)
- Exact term searches (use hybrid_search instead)

Tips:
- Higher similarity scores (closer to 1.0) indicate better matches
- If no results, try broader or different terminology
- For keyword-heavy queries, prefer hybrid_search
"""
    args_schema: Type[BaseModel] = SemanticSearchSchema

    kg: Any = None

    def _run(self, query: str, k: int = 5) -> str:
        """Execute semantic search."""
        if self.kg is None:
            return "ERROR: Knowledge graph not initialized. This is a system configuration issue."

        if not query or not query.strip():
            return "ERROR: Query cannot be empty. Provide a natural language query."

        try:
            results = self.kg.semantic_search(query, k=k)
            if not results:
                return "No results found. The graph may be empty or no similar content exists. Try a different query or add more data to the graph."

            output = []
            for i, doc in enumerate(results, 1):
                output.append(f"[{i}] {doc.page_content}")
                if doc.metadata:
                    score = doc.metadata.get('score')
                    if score is not None:
                        output.append(f"    Similarity: {score:.3f}")

            return "\n".join(output)
        except AttributeError as e:
            return f"ERROR: Semantic search not available - embeddings may not be initialized. Details: {str(e)}"
        except Exception as e:
            return f"ERROR: Semantic search failed: {str(e)}. Try hybrid_search or check if the graph has data."


# =============================================================================
# Hybrid Search Tool
# =============================================================================

class HybridSearchTool(BaseTool):
    """Tool for hybrid search combining vector similarity and fulltext search.

    Best for general queries where both semantic understanding and keyword matching matter.
    This is the recommended default search tool for most queries.
    """
    name: str = "hybrid_search"
    description: str = """Search the knowledge graph using hybrid search (vector + fulltext).
Best for general queries - combines semantic understanding with keyword matching.
Use this as your default search tool. More robust than semantic_search alone.

Good examples:
- "knowledge graph construction algorithms"
- "vector database performance comparison"
- "entity linking techniques 2024"

Tips:
- Combines results from both vector similarity AND keyword matching
- More robust than semantic_search for mixed queries
- Use when query contains both concepts and specific terms
- If results are noisy, try semantic_search for purer semantic matches
"""
    args_schema: Type[BaseModel] = HybridSearchSchema

    kg: Any = None

    def _run(self, query: str, k: int = 5) -> str:
        """Execute hybrid search."""
        if self.kg is None:
            return "ERROR: Knowledge graph not initialized. This is a system configuration issue."

        if not query or not query.strip():
            return "ERROR: Query cannot be empty. Provide a search query."

        try:
            results = self.kg.hybrid_search(query, k=k)
            if not results:
                return "No results found. The graph may be empty. Try adding data first."

            output = []
            for i, doc in enumerate(results, 1):
                output.append(f"[{i}] {doc.page_content}")

            return "\n".join(output)
        except AttributeError as e:
            return f"ERROR: Search not available - {str(e)}. Check if the knowledge graph is properly configured."
        except Exception as e:
            return f"ERROR: Hybrid search failed: {str(e)}. Try a simpler query."


# =============================================================================
# Text-to-Cypher Tool
# =============================================================================

class TextToCypherTool(BaseTool):
    """Tool for converting natural language to Cypher queries.

    Uses FalkorDBQAChain with Mistral LLM for text-to-cypher conversion.
    Best for structured queries about specific patterns and relationships.
    """
    name: str = "text_to_cypher"
    description: str = """Convert natural language to Cypher and query the knowledge graph.
Use when you need structured data about specific patterns, relationships, or when asking
questions about entity counts, connections, or graph structure.

Good examples:
- "What entities are connected to node X?"
- "How many nodes of type Thought exist?"
- "Find all paths between entity A and entity B"
- "List all relationships originating from concept Y"

Tips:
- Requires graph to have data with defined node types
- Best for structural/relational queries, not content search
- Uses Mistral LLM for text-to-Cypher translation
- For content-based searches, prefer semantic_search or hybrid_search
"""
    args_schema: Type[BaseModel] = TextToCypherSchema

    qa_chain: Any = None

    @classmethod
    def create(
        cls,
        graph: FalkorDBGraph,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None
    ) -> "TextToCypherTool":
        """Create a TextToCypherTool with FalkorDBQAChain."""
        llm = ChatMistralAI(model=model, api_key=api_key)
        chain = FalkorDBQAChain.from_llm(
            llm=llm,
            graph=graph,
            allow_dangerous_requests=True,
            verbose=True
        )
        return cls(qa_chain=chain)

    def _run(self, question: str) -> str:
        """Execute text-to-cypher query."""
        if self.qa_chain is None:
            return "ERROR: QA chain not initialized. This requires MISTRAL_API_KEY to be set."

        if not question or not question.strip():
            return "ERROR: Question cannot be empty. Provide a natural language question about the graph."

        try:
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result", "")
            if not answer:
                return "No answer generated. The graph may be empty or the question may not match any data."
            return answer
        except Exception as e:
            return f"ERROR: Text-to-Cypher failed: {str(e)}. Try rephrasing your question or check if the graph has relevant data."


# =============================================================================
# Tool Factory Functions
# =============================================================================

def create_semantic_search_tool(kg) -> SemanticSearchTool:
    """Create a semantic search tool for the given knowledge graph.

    Args:
        kg: KnowledgeGraph instance with semantic_search method

    Returns:
        Configured SemanticSearchTool
    """
    return SemanticSearchTool(kg=kg)


def create_hybrid_search_tool(kg) -> HybridSearchTool:
    """Create a hybrid search tool for the given knowledge graph.

    Args:
        kg: KnowledgeGraph instance with hybrid_search method

    Returns:
        Configured HybridSearchTool
    """
    return HybridSearchTool(kg=kg)


def create_text_to_cypher_tool(
    kg,
    model: str = "mistral-large-latest",
    api_key: Optional[str] = None
) -> TextToCypherTool:
    """Create a text-to-cypher tool for the given knowledge graph.

    Args:
        kg: KnowledgeGraph instance with get_langchain_graph method
        model: Mistral model to use for Cypher generation
        api_key: Mistral API key (falls back to MISTRAL_API_KEY env var)

    Returns:
        Configured TextToCypherTool
    """
    lc_graph = kg.get_langchain_graph()
    return TextToCypherTool.create(graph=lc_graph, model=model, api_key=api_key)