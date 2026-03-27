# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# FalkorDB LLM invocation functions - reuses Neo4j prompts (OpenCypher compatible)

from kgot.controller.neo4j.llm_invocation_base import (
    define_cypher_query_given_new_information_base,
    define_final_solution_base,
    define_math_tool_call_base,
    define_need_for_math_before_parsing_base,
    define_retrieve_query_base,
    define_tool_calls_base,
    fix_cypher_base,
    merge_reasons_to_insert_base,
    parse_solution_with_llm_base,
)

__all__ = [
    "define_cypher_query_given_new_information_base",
    "define_final_solution_base",
    "define_math_tool_call_base",
    "define_need_for_math_before_parsing_base",
    "define_retrieve_query_base",
    "define_tool_calls_base",
    "fix_cypher_base",
    "merge_reasons_to_insert_base",
    "parse_solution_with_llm_base",
]
