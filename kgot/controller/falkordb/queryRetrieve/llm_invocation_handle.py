# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# FalkorDB queryRetrieve LLM invocation handle

from kgot.controller.neo4j.queryRetrieve.llm_invocation_handle import (
    define_cypher_query_given_new_information,
    define_final_solution,
    define_forced_retrieve_queries,
    define_math_tool_call,
    define_need_for_math_before_parsing,
    define_next_step,
    define_retrieve_query,
    define_tool_calls,
    fix_cypher,
    generate_forced_solution,
    merge_reasons_to_insert,
    parse_solution_with_llm,
)

__all__ = [
    "define_cypher_query_given_new_information",
    "define_final_solution",
    "define_forced_retrieve_queries",
    "define_math_tool_call",
    "define_need_for_math_before_parsing",
    "define_next_step",
    "define_retrieve_query",
    "define_tool_calls",
    "fix_cypher",
    "generate_forced_solution",
    "merge_reasons_to_insert",
    "parse_solution_with_llm",
]
