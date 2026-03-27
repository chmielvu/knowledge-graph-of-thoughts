# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# FalkorDB queryRetrieve Controller - uses Cypher queries to retrieve specific data
# Enhanced with semantic search capabilities

import importlib
import json
import os
from typing import List, Tuple, Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler

from kgot.controller import ControllerInterface
from kgot.controller.falkordb.queryRetrieve.llm_invocation_handle import (
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
from kgot.tools.PythonCodeTool import RunPythonCodeTool
from kgot.utils import State
from kgot.utils.tracing import get_langfuse_handler
from kgot.utils.utils import ensure_file_path_exists, is_empty_solution


class Controller(ControllerInterface):
    """FalkorDB queryRetrieve Controller with semantic search capabilities.

    Enhances Cypher-based retrieval with:
    - Semantic search using Mistral embeddings
    - Hybrid search (vector + fulltext)
    - Text-to-Cypher via LangChain FalkorDBQAChain
    """

    def __init__(self,
                 falkordb_host: str = "localhost",
                 falkordb_port: int = 6379,
                 falkordb_username: str | None = None,
                 falkordb_password: str | None = None,
                 falkordb_graph_name: str = "kgot_graph",
                 python_executor_uri: str | None = None,
                 llm_execution_model: str | None = None,
                 llm_execution_temperature: float = 0.0,
                 statistics_file_name: str = "llm_cost.json",
                 db_choice: str = "falkordb",
                 tool_choice: str = "tools_v2_3",

                 # Semantic search configuration
                 mistral_api_key: Optional[str] = None,
                 enable_semantic_search: bool = True,

                 *args, **kwargs) -> None:
        super().__init__(llm_execution_model=llm_execution_model,
                        llm_execution_temperature=llm_execution_temperature,
                        *args, **kwargs)

        ensure_file_path_exists(statistics_file_name)

        # Initialize FalkorDB Knowledge Graph with semantic search
        self.graph = State.knowledge_graph(
            db_choice,
            falkordb_host=falkordb_host,
            falkordb_port=falkordb_port,
            falkordb_username=falkordb_username,
            falkordb_password=falkordb_password,
            graph_name=falkordb_graph_name,
            mistral_api_key=mistral_api_key or os.getenv("MISTRAL_API_KEY"),
            enable_semantic_search=enable_semantic_search,
        )
        self.usage_statistics = State.usage_statistics(statistics_file_name)

        tool_manager = importlib.import_module(f"kgot.tools.{tool_choice}").ToolManager
        self.tool_manager = tool_manager(self.usage_statistics, python_executor_uri=python_executor_uri,
                                         model_name=llm_execution_model, graph=self.graph)
        self.tools = self.tool_manager.get_tools()

        self.tool_names = {curr_tool.name.lower(): curr_tool for curr_tool in self.tools}

        if self.tools:
            self.llm_execution = self.llm_execution.bind_tools(self.tools, tool_choice="required")

        pythonTool = RunPythonCodeTool(try_to_fix=True, times_to_fix=3, model_name=llm_execution_model,
                                        temperature=llm_execution_temperature, python_executor_uri=python_executor_uri,
                                        usage_statistics=self.usage_statistics)
        self.llm_math_executor = self.llm_math_executor.bind_tools([pythonTool], tool_choice="required")

        # Initialize Langfuse tracing (optional - requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
        self.langfuse_handler = get_langfuse_handler()
        self._callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        # Bind callbacks to LLMs for automatic tracing
        if self._callbacks and hasattr(self.llm_planning, 'bind'):
            self.llm_planning = self.llm_planning.bind(config={"callbacks": self._callbacks})
        if self._callbacks and hasattr(self.llm_execution, 'bind'):
            self.llm_execution = self.llm_execution.bind(config={"callbacks": self._callbacks})
        if self._callbacks and hasattr(self.llm_math_executor, 'bind'):
            self.llm_math_executor = self.llm_math_executor.bind(config={"callbacks": self._callbacks})

    def get_callbacks(self) -> list[BaseCallbackHandler]:
        """Get configured callback handlers for LLM tracing."""
        return self._callbacks

    def _iterative_next_step_logic(self, problem: str, *args, **kwargs) -> Tuple[str, int]:
        solution = ''
        raw_solutions = []
        existing_entities_and_relationships = ''
        tool_calls_made = []
        current_iteration = 0

        while current_iteration < self.max_iterations:
            retrieve_next_step = {'RETRIEVE': 0, 'INSERT': 0, 'RETRIEVE_CONTENT': [], 'INSERT_CONTENT': []}

            for i in range(self.num_next_steps_decision):
                retrieve_query, query_type = define_next_step(self.llm_planning, problem,
                                                              existing_entities_and_relationships, tool_calls_made,
                                                              self.usage_statistics)
                try:
                    retrieve_next_step[query_type] += 1
                    retrieve_next_step[query_type + '_CONTENT'].append(retrieve_query)
                except KeyError:
                    self.logger.error(f'Unknown query type: {query_type}')

            if retrieve_next_step['RETRIEVE'] > retrieve_next_step['INSERT']:
                raw_solutions.extend(self._perform_retrieve_branch(problem, existing_entities_and_relationships,
                                              retrieve_next_step['RETRIEVE_CONTENT']))
                break

            reason_to_insert = retrieve_next_step['INSERT_CONTENT'][0] if retrieve_next_step['INSERT'] > 0 else ''
            if retrieve_next_step['INSERT'] > 1:
                reason_to_insert = merge_reasons_to_insert(self.llm_planning, retrieve_next_step['INSERT_CONTENT'], self.usage_statistics)

            existing_entities_and_relationships = self._insert_logic(problem, reason_to_insert, tool_calls_made, existing_entities_and_relationships)
            current_iteration += 1

        solution = self._retrieve_logic(problem, existing_entities_and_relationships, current_iteration, raw_solutions)
        return solution, current_iteration

    def _insert_logic(self, query, reason_to_insert, tool_calls_made, existing_entities_and_relationships, *args, **kwargs):
        tool_calls = define_tool_calls(self.llm_execution, query, existing_entities_and_relationships,
                                       reason_to_insert, tool_calls_made, self.usage_statistics)
        tools_results = self._invoke_tools_after_llm_response(tool_calls)
        tool_calls_made.extend(tool_calls)
        
        for call, result in zip(tool_calls, tools_results):
            new_information = f"function returned: {result}"
            new_information_cypher_queries = define_cypher_query_given_new_information(
                self.llm_planning, query, existing_entities_and_relationships,
                new_information, reason_to_insert, self.usage_statistics)

            for single_query in new_information_cypher_queries:
                write_response = self.graph.write_query(single_query)
                retry_i = 0
                while not write_response[0] and retry_i < self.max_cypher_fixing_retry:
                    retry_i += 1
                    single_query = fix_cypher(self.llm_planning, single_query, write_response[1], self.usage_statistics)
                    write_response = self.graph.write_query(single_query)

            existing_entities_and_relationships = self.graph.get_current_graph_state()
        return existing_entities_and_relationships

    def _retrieve_logic(self, query, existing_entities_and_relationships, current_iteration, solutions):
        if current_iteration == self.max_iterations and len(solutions) == 0:
            retrieve_queries = [define_forced_retrieve_queries(self.llm_planning, query, existing_entities_and_relationships, self.usage_statistics)
                              for _ in range(self.num_next_steps_decision)]
            solutions.extend(self._perform_retrieve_branch(query, existing_entities_and_relationships, retrieve_queries))

        if solutions:
            array_parsed_solutions = []
            for sol in solutions:
                need_math = define_need_for_math_before_parsing(self.llm_planning, query, sol, self.usage_statistics)
                if need_math:
                    sol = self._get_math_response(query, sol)
                for i in range(self.max_final_solution_parsing):
                    array_parsed_solutions.append(parse_solution_with_llm(self.llm_planning, query, sol, self.gaia_formatter, self.usage_statistics))

            if all(not p.strip() for p in array_parsed_solutions if p):
                forced_solution = generate_forced_solution(self.llm_planning, query, existing_entities_and_relationships, self.usage_statistics)
                solution = parse_solution_with_llm(self.llm_planning, query, forced_solution, self.gaia_formatter, self.usage_statistics)
            else:
                solution = define_final_solution(self.llm_planning, query, str(solutions), array_parsed_solutions, self.usage_statistics)
        else:
            forced_solution = generate_forced_solution(self.llm_planning, query, existing_entities_and_relationships, self.usage_statistics)
            solution = parse_solution_with_llm(self.llm_planning, query, forced_solution, self.gaia_formatter, self.usage_statistics)
        return solution

    def _perform_retrieve_branch(self, query, existing_entities_and_relationships, retrieve_queries):
        solutions = []
        if isinstance(retrieve_queries, str):
            retrieve_queries = [retrieve_queries]

        for retrieve_query in retrieve_queries:
            get_result = self.graph.get_query(retrieve_query)
            retrieve_retry_i = 0
            while (not get_result[1] or not get_result[0]) and retrieve_retry_i < self.max_retrieve_query_retry:
                retrieve_retry_i += 1
                fix_retry_i = 0
                while not get_result[1] and fix_retry_i < self.max_cypher_fixing_retry:
                    fix_retry_i += 1
                    retrieve_query = fix_cypher(self.llm_planning, retrieve_query, get_result[2], self.usage_statistics)
                    get_result = self.graph.get_query(retrieve_query)
                if not get_result[1] or not get_result[0]:
                    new_query = define_retrieve_query(self.llm_planning, query, existing_entities_and_relationships, retrieve_query, self.usage_statistics)
                    get_result = self.graph.get_query(new_query)
            solutions.append(get_result[0])
        return solutions

    def _get_math_response(self, query, solution):
        python_tool_call = define_math_tool_call(self.llm_math_executor, query, solution, self.usage_statistics)
        math_solution = self._invoke_tools_after_llm_response(python_tool_call)
        if math_solution and math_solution[0]:
            solution = f"{solution}\nPython result: {math_solution}"
        return solution

    def _invoke_tools_after_llm_response(self, tool_calls):
        outputs = []
        for tool_call in tool_calls:
            tool_name = tool_call['name'].lower()
            tool_args = tool_call['args']
            tool_call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            if tool_call_key in self.tool_call_results_cache:
                tool_output = self.tool_call_results_cache[tool_call_key]
            else:
                selected_tool = self.tool_names.get(tool_name)
                if selected_tool:
                    tool_output = self._invoke_tool_with_retry(selected_tool, tool_args)
                    if tool_name != 'extract_zip':
                        self.tool_call_results_cache[tool_call_key] = tool_output
                else:
                    tool_output = None
            outputs.append(tool_output)
        return outputs
