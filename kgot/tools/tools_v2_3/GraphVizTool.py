# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import os
import time
from typing import Any, Type

from langchain.tools import BaseTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from pyvis.network import Network


class GraphVizSchema(BaseModel):
    limit: int = Field(default=200, description="Maximum number of relationships to render.")
    output_path: str | None = Field(
        default=None,
        description="Optional output HTML path. If omitted, a file under results/graph_viz is created.",
    )


class GraphVizTool(BaseTool):
    name: str = "visualize_graph"
    description: str = (
        "Export the current Neo4j graph into an interactive HTML visualization. "
        "Useful for debugging and inspecting the generated graph."
    )
    args_schema: Type[BaseModel] = GraphVizSchema

    def _run(self, limit: int = 200, output_path: str | None = None) -> str:
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")
        if not uri or not user or not password:
            return "Neo4j connection details are missing. Configure NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD."

        if output_path is None:
            out_dir = os.path.join("results", "graph_viz")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"neo4j_graph_{int(time.time())}.html")

        edge_query = """
        MATCH (n)-[r]->(m)
        RETURN elementId(n) AS source_id, labels(n) AS source_labels, properties(n) AS source_props,
               elementId(m) AS target_id, labels(m) AS target_labels, properties(m) AS target_props,
               type(r) AS rel_type
        LIMIT $limit
        """
        node_query = """
        MATCH (n)
        RETURN elementId(n) AS node_id, labels(n) AS node_labels, properties(n) AS node_props
        LIMIT $limit
        """

        net = Network(
            height="750px",
            width="100%",
            directed=True,
            bgcolor="#ffffff",
            font_color="#222222",
            cdn_resources="in_line",
        )
        added_nodes: set[str] = set()
        edge_count = 0

        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                edge_records = list(session.run(edge_query, limit=limit))
                if edge_records:
                    for record in edge_records:
                        source_id = record["source_id"]
                        target_id = record["target_id"]
                        if source_id not in added_nodes:
                            net.add_node(source_id, label=self._node_label(record["source_labels"], record["source_props"]))
                            added_nodes.add(source_id)
                        if target_id not in added_nodes:
                            net.add_node(target_id, label=self._node_label(record["target_labels"], record["target_props"]))
                            added_nodes.add(target_id)
                        net.add_edge(source_id, target_id, label=record["rel_type"])
                        edge_count += 1
                else:
                    for record in session.run(node_query, limit=limit):
                        node_id = record["node_id"]
                        if node_id not in added_nodes:
                            net.add_node(node_id, label=self._node_label(record["node_labels"], record["node_props"]))
                            added_nodes.add(node_id)

        net.write_html(output_path)
        return (
            f"Graph visualization created at {os.path.abspath(output_path)}. "
            f"Nodes rendered: {len(added_nodes)}. Edges rendered: {edge_count}."
        )

    def _node_label(self, labels: list[str], props: dict[str, Any]) -> str:
        primary = (
            props.get("name")
            or props.get("title")
            or props.get("label")
            or props.get("problem")
            or props.get("query")
        )
        label_prefix = labels[0] if labels else "Node"
        if primary:
            return f"{label_prefix}: {str(primary)[:80]}"
        return label_prefix
