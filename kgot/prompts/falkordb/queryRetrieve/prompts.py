# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# FalkorDB queryRetrieve prompts with semantic search awareness

from kgot.prompts.neo4j.queryRetrieve.prompts import (
    DEFINE_FORCED_RETRIEVE_QUERY_TEMPLATE,
    DEFINE_FORCED_SOLUTION_TEMPLATE,
)

DEFINE_NEXT_STEP_PROMPT_TEMPLATE = """
<task>
You are a problem solver using a FalkorDB knowledge graph with multiple search capabilities:
1. **Cypher queries** - For structured data retrieval (exact patterns, relationships, counts)
2. **Semantic search** - For conceptually similar content (when you don't know exact keywords)
3. **Hybrid search** - For general queries combining keywords and concepts
4. **Text-to-Cypher** - For natural language to graph queries
Note that the database may be incomplete.
</task>

<instructions>
Understand the initial problem, the initial problem nuances, *ALL the existing data* in the database and the tools already called.
Can you solve the initial problem using the existing data in the database?

- **If data exists and is sufficient**: Return a Cypher query to retrieve the necessary data. Set query_type to RETRIEVE.
- **If data is insufficient**: Return why you couldn't solve it and what's missing. Set query_type to INSERT.
- **For semantic/conceptual searches**: Use semantic_search tool instead of Cypher when looking for similar ideas.
- **For general searches**: Use hybrid_search tool for best results combining keywords and concepts.

Tool Selection Guide:
- Need specific entities/relationships? → Cypher query (RETRIEVE)
- Looking for similar concepts? → semantic_search tool (INSERT first, then search)
- General exploration? → hybrid_search tool (INSERT first, then search)
- Structured questions about graph? → text_to_cypher tool
</instructions>

<examples>

<examples_retrieve>
<example_retrieve_1>
Initial problem: Retrieve all books written by "J.K. Rowling".
Existing entities: Author: [{{name: "J.K. Rowling", author_id: "A1"}}, {{name: "George R.R. Martin", author_id: "A2"}}], Book: [{{title: "Harry Potter and the Philosopher's Stone", book_id: "B1"}}, {{title: "Harry Potter and the Chamber of Secrets", book_id: "B2"}}, {{title: "A Game of Thrones", book_id: "B3"}}]
Existing relationships: (A1)-[:WROTE]->(B1), (A1)-[:WROTE]->(B2), (A2)-[:WROTE]->(B3)
Solution:
query: '
MATCH (a:Author {{name: "J.K. Rowling"}})-[:WROTE]->(b:Book)
RETURN b.title AS book_title
'
query_type: RETRIEVE
</example_retrieve_1>
<example_retrieve_2>
Initial problem: Find concepts similar to "machine learning" in the knowledge graph.
Existing entities: Thought: [{{content: "Neural networks are a type of ML model"}}, {{content: "Deep learning uses layered neural networks"}}]
Solution:
query: 'Use semantic_search tool with query "machine learning" to find conceptually similar content'
query_type: RETRIEVE
</example_retrieve_2>
</examples_retrieve>

<examples_insert>
<example_insert_1>
Initial problem: Retrieve all books written by "J.K. Rowling".
Existing entities: {{name: "George R.R. Martin", author_id: "A2"}}], Book: [{{title: "A Game of Thrones", book_id: "B3"}}]
Existing relationships: (A2)-[:WROTE]->(B3)
Solution:
query: 'There are no books of "J.K. Rowling" in the current database, we need more'
query_type: INSERT
</example_insert_1>
<example_insert_2>
Initial problem: Find information about quantum computing research
Existing entities: []
Existing relationships: []
Solution:
query: 'The given database is empty, we need to populate it with quantum computing information first'
query_type: INSERT
</example_insert_2>
</examples_insert>

</examples>

<initial_problem>
{initial_query}
</initial_problem>

<existing_data>
{existing_entities_and_relationships}
</existing_data>

<tool_calls_made>
{tool_calls_made}
</tool_calls_made>
"""

__all__ = [
    "DEFINE_FORCED_RETRIEVE_QUERY_TEMPLATE",
    "DEFINE_FORCED_SOLUTION_TEMPLATE",
    "DEFINE_NEXT_STEP_PROMPT_TEMPLATE",
]