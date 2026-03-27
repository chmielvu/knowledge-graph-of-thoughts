# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# FalkorDB base prompts - delegates to Neo4j prompts (OpenCypher compatible)
#
# Note: FalkorDB uses OpenCypher which is compatible with Neo4j's Cypher.
# The only difference is ID() vs elementId() which is handled in the 
# knowledge_graph layer, not in prompts.

# Re-export all prompts from Neo4j
from kgot.prompts.neo4j.base_prompts import (
    DEFINE_CYPHER_QUERY_GIVEN_NEW_INFORMATION_PROMPT_TEMPLATE,
    DEFINE_MATH_TOOL_CALL_PROMPT_TEMPLATE,
    DEFINE_NEED_FOR_MATH_PROMPT_TEMPLATE,
    DEFINE_REASON_TO_INSERT_PROMPT_TEMPLATE,
    DEFINE_RETRIEVE_QUERY_PROMPT_TEMPLATE,
    DEFINE_TOOL_CALLS_PROMPT_TEMPLATE,
    FIX_CYPHER_PROMPT_TEMPLATE,
    PARSE_FINAL_SOLUTION_WITH_LLM_PROMPT_TEMPLATE,
    get_formatter,
)

__all__ = [
    "DEFINE_CYPHER_QUERY_GIVEN_NEW_INFORMATION_PROMPT_TEMPLATE",
    "DEFINE_MATH_TOOL_CALL_PROMPT_TEMPLATE",
    "DEFINE_NEED_FOR_MATH_PROMPT_TEMPLATE",
    "DEFINE_REASON_TO_INSERT_PROMPT_TEMPLATE",
    "DEFINE_RETRIEVE_QUERY_PROMPT_TEMPLATE",
    "DEFINE_TOOL_CALLS_PROMPT_TEMPLATE",
    "FIX_CYPHER_PROMPT_TEMPLATE",
    "PARSE_FINAL_SOLUTION_WITH_LLM_PROMPT_TEMPLATE",
    "get_formatter",
]
