"""
Deep Research State

State definitions for the LangGraph research agent workflow.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from .configuration import DeepResearchConfig


class Source(BaseModel):
    """A research source consulted during investigation."""
    title: str = Field(description="Title of the source page")
    url: str = Field(description="URL of the source")
    content: str = Field(default="", description="Extracted content from source")
    raw_content: Optional[str] = Field(
        default=None,
        description="Full raw content if available"
    )
    source_type: str = Field(
        default="web",
        description="Type: web, wikipedia, ai_search"
    )
    credibility_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Credibility score 0-1"
    )
    accessed_at: datetime = Field(
        default_factory=datetime.now,
        description="When this source was accessed"
    )


class ResearchGap(BaseModel):
    """An area that needs more research."""
    topic: str = Field(description="The topic needing more research")
    reason: str = Field(description="Why this area needs more investigation")


class SearchQuery(BaseModel):
    """A search query with metadata."""
    query: str = Field(description="The search query")
    source: str = Field(
        default="generated",
        description="How this query was generated"
    )
    results_count: int = Field(
        default=0,
        description="Number of results found"
    )


class ResearchState(BaseModel):
    """
    State for the research agent workflow.

    Tracks the entire research process including:
    - Original topic and generated queries
    - Gathered sources
    - Synthesized findings
    - Identified gaps
    - Loop count for termination
    """

    # Research topic
    research_topic: str = Field(description="The original research question/topic")

    # Query management
    search_query: str = Field(
        default="",
        description="Current search query being executed"
    )
    queries_generated: List[SearchQuery] = Field(
        default_factory=list,
        description="All queries generated during research"
    )

    # Sources
    sources_gathered: List[Source] = Field(
        default_factory=list,
        description="All sources consulted during research"
    )

    # Research results
    web_research_results: str = Field(
        default="",
        description="Raw text of web research results"
    )
    summary: str = Field(
        default="",
        description="Current synthesized summary"
    )

    # Gaps and next steps
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="Topics needing further research"
    )

    # Loop tracking
    research_loop_count: int = Field(
        default=0,
        ge=0,
        description="Number of research iterations completed"
    )

    # Completion status
    is_complete: bool = Field(
        default=False,
        description="Whether research is complete"
    )

    # Configuration reference (not serialized)
    config: Optional[DeepResearchConfig] = Field(
        default=None,
        description="Research configuration",
        exclude=True
    )

    def add_source(self, source: Source) -> None:
        """Add a source to the gathered list."""
        # Avoid duplicates
        if not any(s.url == source.url for s in self.sources_gathered):
            self.sources_gathered.append(source)

    def add_query(self, query: str, source: str = "generated") -> None:
        """Add a search query to the list."""
        self.queries_generated.append(SearchQuery(
            query=query,
            source=source
        ))

    def increment_loop(self) -> None:
        """Increment the research loop counter."""
        self.research_loop_count += 1

    def should_continue(self) -> bool:
        """
        Determine if research should continue.

        Returns False if:
        - Max loops reached
        - Marked as complete
        - No gaps identified
        """
        if self.is_complete:
            return False
        if self.config and self.research_loop_count >= self.config.max_research_loops:
            return False
        if not self.gaps_identified and self.research_loop_count > 0:
            return False
        return True


class SummaryState(BaseModel):
    """
    State for the summarization step.

    Captures the summary and any follow-up needs.
    """
    research_topic: str
    sources: List[Source]
    web_research_results: str
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReflectState(BaseModel):
    """
    State for the reflection step.

    Captures gaps and next steps.
    """
    research_topic: str
    summary: str
    sources: List[Source]
    gaps: List[str] = Field(default_factory=list)
    is_sufficient: bool = False
    next_query: str = ""