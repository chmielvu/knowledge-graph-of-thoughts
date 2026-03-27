# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Main authors: Lorenzo Paleari

from kgot.tools.PythonCodeTool import RunPythonCodeTool
from kgot.tools.tool_manager_interface import ToolManagerInterface
from kgot.tools.tools_v2_3.ExtractZipTool import ExtractZipTool
from kgot.tools.tools_v2_3.FalkorDBSearchTool import (
    create_hybrid_search_tool,
    create_semantic_search_tool,
    create_text_to_cypher_tool,
)
from kgot.tools.tools_v2_3.GraphVizTool import GraphVizTool
try:
    from kgot.tools.tools_v2_3.ImageQuestionTool import ImageQuestionTool
except Exception:
    ImageQuestionTool = None
from kgot.tools.tools_v2_3.LLMTool import LangchainLLMTool
from kgot.tools.tools_v2_3.PollinationsSearchTool import PollinationsSearchTool
from kgot.tools.tools_v2_3.web_browser import WebBrowserTool, init_browser
from kgot.tools.tools_v2_3.TextInspectorTool import TextInspectorTool
from kgot.tools.tools_v2_3.TongyiuDeepResearch import TongyiuDeepResearchTool
from kgot.utils import UsageStatistics
import os


class ToolManager(ToolManagerInterface):
    """
    ToolManager v2.3 class for managing tools.
    Inherits from ToolManagerInterface.

    Attributes:
        usage_statistics (UsageStatistics): Usage statistics for the tools
        config_path (str): Path to the configuration file
        tools (list[BaseTool]): List of tools
        graph: Optional knowledge graph instance for FalkorDB search tools
    """

    def __init__(
        self,
        usage_statistics: UsageStatistics,
        base_config_path: str = "kgot/config_tools.json",
        additional_config_path: str = "kgot/tools/tools_v2_3/additional_config_tools.template.json",
        python_executor_uri: str = "http://localhost:16000",
        model_name: str = "gpt-4o-mini",
        graph=None,
    ) -> None:
        """
        Initialize the ToolManager.

        Args:
            usage_statistics (UsageStatistics): Usage statistics for the tools
            base_config_path (str): Path to the configuration file
            additional_config_path (str): Path to the additional configuration file
            python_executor_uri (str): URI for the Python Docker service
            model_name (str): Model name to use for tools that need LLM
            graph: Optional knowledge graph instance for FalkorDB search tools
        """
        super().__init__(usage_statistics, base_config_path, additional_config_path)

        self.graph = graph
        init_browser()
        ### TOOLS ###
        extract_zip_tool = ExtractZipTool()
        web_browser_tool = WebBrowserTool(model_name=model_name, temperature=0.5, usage_statistics=usage_statistics)
        pollinations_search_tool = PollinationsSearchTool(usage_statistics=usage_statistics)
        graph_viz_tool = GraphVizTool()
        LLM_tool = LangchainLLMTool(model_name=model_name, temperature=0.5, usage_statistics=usage_statistics)
        textInspectorTool = TextInspectorTool(model_name=model_name, temperature=0.5, usage_statistics=usage_statistics)
        image_question_tool = ImageQuestionTool(model_name=model_name, temperature=0.5, usage_statistics=usage_statistics) if ImageQuestionTool is not None else None
        run_python_tool = RunPythonCodeTool(
            try_to_fix=True,
            times_to_fix=3,
            model_name=model_name,
            temperature=0.5,
            python_executor_uri=python_executor_uri,
            usage_statistics=usage_statistics,
        )

        self.tools.extend([
            LLM_tool,
            textInspectorTool,
            web_browser_tool,
            pollinations_search_tool,
            run_python_tool,
            extract_zip_tool,
            graph_viz_tool,
        ])
        if image_question_tool is not None:
            self.tools.append(image_question_tool)

        # Deep research tool (Tongyi-DeepResearch with Crawl4AI)
        try:
            deep_research_tool = TongyiuDeepResearchTool()
            self.tools.append(deep_research_tool)
        except Exception as e:
            print(f"Warning: Could not initialize TongyiuDeepResearchTool: {e}")

        # FalkorDB Search Tools (if graph is provided and supports semantic search)
        if self.graph is not None:
            self._initialize_falkordb_search_tools()

        # Test for python docker
        self._test_python_container(run_python_tool)

    def _test_python_container(self, python_tool: RunPythonCodeTool) -> None:
        """
        Test the Python Docker service by running a simple code snippet.
        If the service is not running, an exception is raised.

        Args:
            python_executor_uri (str): URI for the Python Docker service
        """
        response = python_tool._run("""
import os
print("Hello, World!")
print("Python Docker service is running.")
""")

        if response.get("error"):
            print(
                "\n\n\033[1;31m" + "Failed to connect to Docker instance! Be sure to have a running Docker instance and double check the connection parameters.\n\n")
            exit(1)

    def _initialize_falkordb_search_tools(self) -> None:
        """
        Initialize FalkorDB search tools if the graph supports them.

        Creates:
        - semantic_search: Vector similarity search
        - hybrid_search: Vector + fulltext search
        - text_to_cypher: Natural language to Cypher (requires Mistral API key)
        """
        try:
            # Check if graph has semantic search capability
            if not hasattr(self.graph, 'semantic_search'):
                print("Info: Graph does not support semantic search. Skipping FalkorDB search tools.")
                return

            # Create semantic search tool
            semantic_tool = create_semantic_search_tool(self.graph)
            self.tools.append(semantic_tool)
            print("Info: FalkorDB semantic search tool initialized.")

            # Create hybrid search tool
            hybrid_tool = create_hybrid_search_tool(self.graph)
            self.tools.append(hybrid_tool)
            print("Info: FalkorDB hybrid search tool initialized.")

            # Create text-to-cypher tool (requires Mistral API key)
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            if mistral_api_key:
                try:
                    text_to_cypher_tool = create_text_to_cypher_tool(
                        self.graph,
                        model="mistral-large-latest",
                        api_key=mistral_api_key
                    )
                    self.tools.append(text_to_cypher_tool)
                    print("Info: FalkorDB text-to-cypher tool initialized.")
                except Exception as e:
                    print(f"Warning: Could not initialize text-to-cypher tool: {e}")
            else:
                print("Info: MISTRAL_API_KEY not set. Skipping text-to-cypher tool.")

        except Exception as e:
            print(f"Warning: Could not initialize FalkorDB search tools: {e}")
            # Don't fail completely if FalkorDB tools fail to initialize
            print("Info: Continuing without FalkorDB search tools.")


if __name__ == "__main__":
    tool_manager = ToolManager(None, additional_config_path="kgot/tools/tools_v2_3/additional_config_tools.template.json")
    print(tool_manager.get_tools())
