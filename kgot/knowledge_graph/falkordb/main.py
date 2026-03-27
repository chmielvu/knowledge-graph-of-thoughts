# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Main authors: Lorenzo Paleari (original Neo4j), FalkorDB adaptation
# Enhanced with Mistral embeddings and LangChain integration
#
# FalkorDB Knowledge Graph Backend for KGoT
#
# Key features:
# - Native multi-tenancy via graph names
# - Cypher query support (OpenCypher compatible)
# - Vector search with Mistral embeddings (mistral-embed)
# - Hybrid search (vector + fulltext) via LangChain FalkorDBVector
# - GraphRAG-SDK compatibility

import json
import logging
import os
from typing import Any, List, Tuple, Optional, Dict

from falkordb import FalkorDB

from kgot.knowledge_graph.kg_interface import KnowledgeGraphInterface

# LangChain integrations
from langchain_community.graphs import FalkorDBGraph
from langchain_community.vectorstores.falkordb_vector import FalkorDBVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Mistral AI embeddings
try:
    from langchain_mistralai import MistralAIEmbeddings
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    MistralAIEmbeddings = None  # type: ignore


class KnowledgeGraph(KnowledgeGraphInterface):
    """
    A class to interact with a FalkorDB graph database.

    FalkorDB is a Redis-based graph database that supports OpenCypher queries.
    It provides native multi-tenancy - each graph is isolated by name.

    Enhanced with:
    - Mistral AI embeddings (mistral-embed) for semantic search
    - LangChain FalkorDBVector for hybrid search (vector + fulltext)
    - GraphRAG-SDK compatibility

    Attributes:
        db (FalkorDB): The FalkorDB client connection.
        graph: The selected graph instance.
        graph_name (str): The name of the current graph (multi-tenancy).
        embeddings (Embeddings): The embedding model (MistralAIEmbeddings).
        vector_store (FalkorDBVector): LangChain vector store for semantic search.
        lc_graph (FalkorDBGraph): LangChain graph wrapper for text-to-cypher.
        current_folder_name (str): Directory for storing snapshots.
        current_snapshot_id (int): Current snapshot counter.
    """

    # Default configuration
    DEFAULT_EMBEDDING_MODEL = "mistral-embed"
    DEFAULT_NODE_LABEL = "Thought"
    DEFAULT_CONTENT_PROPERTY = "content"
    DEFAULT_EMBEDDING_PROPERTY = "embedding"
    MISTRAL_EMBEDDING_DIMENSION = 1024  # mistral-embed dimension

    def __init__(
        self,
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_username: Optional[str] = None,
        falkordb_password: Optional[str] = None,
        graph_name: str = "kgot_graph",
        ssl: bool = False,

        # Embedding configuration (for semantic search)
        mistral_api_key: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        custom_embeddings: Optional[Embeddings] = None,

        # Vector store configuration
        node_label: str = DEFAULT_NODE_LABEL,
        content_property: str = DEFAULT_CONTENT_PROPERTY,
        embedding_property: str = DEFAULT_EMBEDDING_PROPERTY,

        # Search configuration
        enable_semantic_search: bool = True,
    ) -> None:
        """
        Initialize the FalkorDB KnowledgeGraph.

        Args:
            falkordb_host: FalkorDB server host (default: localhost)
            falkordb_port: FalkorDB server port (default: 6379, Redis protocol)
            falkordb_username: Optional username for authentication
            falkordb_password: Optional password for authentication
            graph_name: Name of the graph to use (default: kgot_graph)
                This enables multi-tenancy - each graph is isolated.
            ssl: Whether to use SSL connection

            mistral_api_key: API key for Mistral AI embeddings
            embedding_model: Embedding model name (default: mistral-embed)
            custom_embeddings: Custom embeddings instance (overrides mistral)

            node_label: Label for nodes in vector search (default: Thought)
            content_property: Property containing text content (default: content)
            embedding_property: Property for storing embeddings (default: embedding)

            enable_semantic_search: Enable semantic search capabilities
        """
        super().__init__(logger_name=f"Controller.{self.__class__.__name__}")

        # Store configuration
        self._host = falkordb_host
        self._port = falkordb_port
        self._graph_name = graph_name
        self._username = falkordb_username
        self._password = falkordb_password
        self._ssl = ssl

        # Embedding configuration
        self._embedding_model = embedding_model
        self._node_label = node_label
        self._content_property = content_property
        self._embedding_property = embedding_property
        self._enable_semantic = enable_semantic_search

        # Initialize FalkorDB connection
        try:
            self.db = FalkorDB(
                host=falkordb_host,
                port=falkordb_port,
                username=falkordb_username,
                password=falkordb_password,
                ssl=ssl
            )
            self.graph = self.db.select_graph(graph_name)
            self._test_connection()
        except Exception as e:
            self.logger.error(f"Failed to connect to FalkorDB: {e}")
            raise ConnectionError(
                f"Failed to connect to FalkorDB at {falkordb_host}:{falkordb_port}. "
                f"Ensure FalkorDB is running. Error: {e}"
            )

        # Initialize embeddings
        self._embeddings: Optional[Embeddings] = None
        self._vector_store: Optional[FalkorDBVector] = None
        self._lc_graph: Optional[FalkorDBGraph] = None
        self._indexes_created = False

        if enable_semantic_search:
            self._init_embeddings(mistral_api_key, custom_embeddings)

        # Auto-create indexes for optimal performance
        self._auto_create_indexes = os.getenv("FALKORDB_AUTO_INDEX", "true").lower() == "true"
        if self._auto_create_indexes:
            try:
                self.ensure_indexes()
            except Exception as e:
                self.logger.warning(f"Auto-index creation failed (non-fatal): {e}")

        # Snapshot tracking
        self.current_folder_name = ""
        self.current_snapshot_id = 0

        self.logger.info(f"Connected to FalkorDB graph '{graph_name}'")

    def _test_connection(self) -> None:
        """
        Test the connection by executing a simple query.
        """
        try:
            result = self.graph.ro_query("RETURN 1 AS test")
            if not result.result_set:
                raise ConnectionError("Connection test returned empty result")
            self.logger.info("FalkorDB connection test successful")
        except Exception as e:
            raise ConnectionError(f"FalkorDB connection test failed: {e}")

    def _init_embeddings(
        self,
        mistral_api_key: Optional[str],
        custom_embeddings: Optional[Embeddings]
    ) -> None:
        """Initialize the embedding model for semantic search."""
        if custom_embeddings is not None:
            self._embeddings = custom_embeddings
            self.logger.info("Using custom embeddings instance")
        elif MISTRAL_AVAILABLE:
            api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
            if api_key:
                self._embeddings = MistralAIEmbeddings(  # type: ignore
                    model=self._embedding_model,
                    api_key=api_key
                )
                self.logger.info(f"Initialized Mistral embeddings with model '{self._embedding_model}'")
            else:
                self.logger.warning(
                    "MISTRAL_API_KEY not set. Semantic search disabled. "
                    "Set MISTRAL_API_KEY environment variable or pass mistral_api_key parameter."
                )
        else:
            self.logger.warning(
                "langchain-mistralai not installed. Install with: pip install langchain-mistralai"
            )

    # =========================================================================
    # LangChain Integration
    # =========================================================================

    def get_langchain_graph(self) -> FalkorDBGraph:
        """
        Get LangChain FalkorDBGraph wrapper.

        Useful for FalkorDBQAChain (text-to-cypher).
        """
        if self._lc_graph is None:
            self._lc_graph = FalkorDBGraph(
                database=self._graph_name,
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                ssl=self._ssl
            )
        return self._lc_graph

    def get_vector_store(self) -> Optional[FalkorDBVector]:
        """
        Get or create LangChain FalkorDBVector for semantic search.

        Returns None if embeddings not initialized.
        """
        if self._vector_store is None and self._embeddings is not None:
            try:
                self._vector_store = FalkorDBVector(
                    embedding=self._embeddings,
                    database=self._graph_name,
                    host=self._host,
                    port=self._port,
                    username=self._username,
                    password=self._password,
                    ssl=self._ssl,
                    node_label=self._node_label,
                    embedding_node_property=self._embedding_property,
                    text_node_property=self._content_property,
                )
                self.logger.info("Connected to FalkorDBVector store")
            except Exception as e:
                self.logger.warning(f"Could not connect to vector store: {e}")
        return self._vector_store

    # =========================================================================
    # Semantic Search Methods
    # =========================================================================

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform semantic (vector similarity) search.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of Document objects with page_content and metadata
        """
        vector_store = self.get_vector_store()
        if vector_store is None:
            self.logger.warning("Vector store not initialized. Cannot perform semantic search.")
            return []
        return vector_store.similarity_search(query, k=k)

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform hybrid search (vector + fulltext fusion).

        Combines semantic similarity with keyword matching for better results.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of Document objects
        """
        vector_store = self.get_vector_store()
        if vector_store is None:
            self.logger.warning("Vector store not initialized. Falling back to fulltext search.")
            return self.fulltext_search_documents(query, k)
        return vector_store.similarity_search(query, k=k, search_type="hybrid")

    def fulltext_search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform fulltext search and return as Documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of Document objects
        """
        results = self.fulltext_search(self._node_label, query, k)
        documents = []
        for result in results:
            if isinstance(result, list) and len(result) >= 2:
                node, score = result[0], result[1]
                if isinstance(node, dict):
                    documents.append(Document(
                        page_content=node.get('properties', {}).get(self._content_property, str(node)),
                        metadata={'score': score, 'id': node.get('id')}
                    ))
        return documents

    def add_thought(
        self,
        thought_id: str,
        content: str,
        thought_type: str = None,
        metadata: Dict = None,
        auto_embed: bool = True
    ) -> None:
        """
        Add a thought node with optional automatic embedding.

        Args:
            thought_id: Unique identifier for the thought
            content: The text content of the thought
            thought_type: Type/label of the thought (default: node_label)
            metadata: Additional properties to set on the node
            auto_embed: Whether to generate and store embedding
        """
        thought_type = thought_type or self._node_label
        props: Dict[str, Any] = {
            "id": thought_id,
            self._content_property: content,
        }
        if metadata:
            props.update(metadata)

        # Create node with Cypher
        prop_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
        cypher = f"""
        MERGE (n:{thought_type} {{id: $id}})
        SET n += {{{prop_str}}}
        """

        try:
            self.graph.query(cypher, props)  # type: ignore
            self.logger.debug(f"Created node {thought_id}")
        except Exception as e:
            self.logger.error(f"Failed to create node: {e}")
            raise

        # Add embedding if enabled
        if auto_embed and self._embeddings:
            try:
                embedding = self._embeddings.embed_query(content)
                embed_cypher = f"""
                MATCH (n:{thought_type} {{id: $id}})
                SET n.{self._embedding_property} = $embedding
                """
                self.graph.query(embed_cypher, {
                    'id': thought_id,
                    'embedding': embedding
                })
                self.logger.debug(f"Added embedding to node {thought_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add embedding: {e}")

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _query_database(self, cypher_query: str, params: dict = None) -> List:
        """
        Execute a Cypher query and return results.

        Args:
            cypher_query: The Cypher query to execute
            params: Optional query parameters

        Returns:
            List of result records
        """
        params = params or {}
        result = self.graph.query(cypher_query, params)
        return result.result_set if result.result_set else []

    def _ro_query_database(self, cypher_query: str, params: dict = None) -> List:
        """
        Execute a read-only Cypher query (optimized for reads).

        Args:
            cypher_query: The Cypher query to execute
            params: Optional query parameters

        Returns:
            List of result records
        """
        params = params or {}
        result = self.graph.ro_query(cypher_query, params)
        return result.result_set if result.result_set else []

    def _create_folder(self, index: int, snapshot_subdir: str = "") -> None:
        """
        Create a folder for storing snapshots.
        """
        folder_name = ""
        if snapshot_subdir:
            folder_name = f"{snapshot_subdir}/"
        folder_name += f"snapshot_{index}"
        self.current_folder_name = folder_name

        folder_dir = os.path.join("./kgot/knowledge_graph/_snapshots", folder_name)
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

    def _export_db(self) -> None:
        """
        Export the current graph state to a JSON file.

        Note: FalkorDB does not have APOC like Neo4j, so we implement
        export using Cypher queries to collect all nodes and edges.
        """
        export_file = f"snapshot_{self.current_snapshot_id}.json"

        # Get all nodes
        nodes_query = """
            MATCH (n)
            RETURN {
                id: ID(n),
                labels: labels(n),
                properties: properties(n)
            } AS node
        """
        nodes_result = self._ro_query_database(nodes_query)

        # Get all relationships
        rels_query = """
            MATCH (source)-[r]->(target)
            RETURN {
                id: ID(r),
                type: type(r),
                source_id: ID(source),
                target_id: ID(target),
                properties: properties(r)
            } AS relationship
        """
        rels_result = self._ro_query_database(rels_query)

        # Build export data - handle result format
        nodes = []
        for n in nodes_result:
            if isinstance(n, list) and len(n) > 0:
                nodes.append(n[0] if isinstance(n[0], dict) else n)
            elif isinstance(n, dict):
                nodes.append(n)

        relationships = []
        for r in rels_result:
            if isinstance(r, list) and len(r) > 0:
                relationships.append(r[0] if isinstance(r[0], dict) else r)
            elif isinstance(r, dict):
                relationships.append(r)

        export_data = {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "graph_name": self._graph_name,
                "snapshot_id": self.current_snapshot_id
            }
        }

        # Write to file
        export_path = os.path.join(
            "./kgot/knowledge_graph/_snapshots",
            self.current_folder_name,
            export_file
        )
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Exported graph to {export_file}")
        self.current_snapshot_id += 1

    # =========================================================================
    # KnowledgeGraphInterface Implementation
    # =========================================================================

    def init_db(
        self,
        snapshot_index: int = 0,
        snapshot_subdir: str = "",
        *args,
        **kwargs
    ) -> None:
        """
        Initialize the database by clearing all nodes and relationships.

        Creates a folder for storing snapshots.
        """
        # Clear the graph using DETACH DELETE
        clear_query = "MATCH (n) DETACH DELETE n"
        self.graph.query(clear_query)
        self.logger.info("Cleared all nodes and relationships")

        # Create snapshot folder
        self._create_folder(snapshot_index, snapshot_subdir)
        self.current_snapshot_id = 0

    def get_current_graph_state(self, *args, **kwargs) -> str:
        """
        Get the current state of the graph as a formatted string.

        Returns all nodes grouped by label and all relationships grouped by type.
        Uses FalkorDB's ID() function instead of Neo4j's elementId().
        """
        # Get nodes grouped by labels
        nodes_query = """
            MATCH (n)
            WITH labels(n) AS labels, collect({
                id: ID(n),
                properties: properties(n)
            }) AS nodes
            RETURN {labels: labels, nodes: nodes} AS groupedNodes
        """
        nodes = self._ro_query_database(nodes_query)
        nodes = nodes if nodes else []

        # Get relationships grouped by type
        rels_query = """
            MATCH (source)-[r]->(target)
            WITH type(r) AS rel_type, collect({
                properties: properties(r),
                source_labels: labels(source),
                target_labels: labels(target),
                source_id: ID(source),
                target_id: ID(target)
            }) AS rels
            RETURN {type: rel_type, rels: rels} AS groupedRels
        """
        rels = self._ro_query_database(rels_query)
        rels = rels if rels else []

        # Format output string (matching Neo4j format for compatibility)
        output = "This is the current state of the FalkorDB database.\n"

        # Format nodes
        output += "Nodes:\n"
        for group in nodes:
            # Handle result format
            group_data = group
            if isinstance(group, list) and len(group) > 0:
                group_data = group[0]
            if isinstance(group_data, dict) and 'groupedNodes' in group_data:
                group_data = group_data['groupedNodes']

            if not isinstance(group_data, dict):
                continue

            labels = group_data.get('labels', [''])
            label = labels[0] if labels else ''
            output += f"  Label: {label}\n"

            for node in group_data.get('nodes', []):
                if not isinstance(node, dict):
                    continue
                node_id = node.get('id', 'unknown')
                properties = node.get('properties', {})
                output += f"    {{id:{node_id}, properties:{properties}}}\n"

        if not nodes:
            output += "  No nodes found\n"

        # Format relationships
        output += "Relationships:\n"
        for group in rels:
            # Handle result format
            group_data = group
            if isinstance(group, list) and len(group) > 0:
                group_data = group[0]
            if isinstance(group_data, dict) and 'groupedRels' in group_data:
                group_data = group_data['groupedRels']

            if not isinstance(group_data, dict):
                continue

            rel_type = group_data.get('type', '')
            output += f"  Type: {rel_type}\n"

            for rel in group_data.get('rels', []):
                if not isinstance(rel, dict):
                    continue
                source_labels = rel.get('source_labels', [''])
                target_labels = rel.get('target_labels', [''])
                source_label = source_labels[0] if source_labels else ''
                target_label = target_labels[0] if target_labels else ''
                source_id = rel.get('source_id', 'unknown')
                target_id = rel.get('target_id', 'unknown')
                properties = rel.get('properties', {})

                output += (
                    f"    {{source: {{id: {source_id}, label: {source_label}}}, "
                    f"target: {{id: {target_id}, label: {target_label}}}, "
                    f"properties: {properties}}}\n"
                )

        if not rels:
            output += "  No relationships found\n"

        return output

    def get_query(
        self,
        query: str,
        *args,
        **kwargs
    ) -> Tuple[str, bool, Exception]:
        """
        Execute a read query on the graph.

        Args:
            query: Cypher query to execute

        Returns:
            Tuple of (result, success, exception)
        """
        if query is None:
            return None, False, ValueError("Query to execute is None")  # type: ignore

        try:
            # Use read-only query for better performance
            result = self._ro_query_database(query)
            return result, True, None  # type: ignore
        except Exception as e:
            self.logger.warning(f"Query failed: {e}")
            return None, False, e  # type: ignore

    def write_query(
        self,
        query: str,
        *args,
        **kwargs
    ) -> Tuple[bool, Exception]:
        """
        Execute a write query on the graph.

        Args:
            query: Cypher query to execute (CREATE, MERGE, DELETE, etc.)

        Returns:
            Tuple of (success, exception)
        """
        if query is None:
            return False, ValueError("Query to execute is None")

        try:
            self.graph.query(query)
            # Export snapshot after each write
            self._export_db()
            return True, None  # type: ignore
        except Exception as e:
            self.logger.error(f"Write query failed: {e}")
            return False, e

    # =========================================================================
    # Additional FalkorDB-Specific Methods
    # =========================================================================

    def get_schema(self) -> dict:
        """
        Get the graph schema (labels, relationship types, properties).

        Returns:
            Dictionary with node_props, rel_props, and relationships
        """
        # Get node properties
        node_props_query = """
            MATCH (n)
            WITH keys(n) AS keys, labels(n) AS labels
            UNWIND labels AS label
            WITH label, collect(DISTINCT key) AS props
            RETURN {label: label, properties: props} AS output
        """
        node_props = self._ro_query_database(node_props_query)

        # Get relationship properties
        rel_props_query = """
            MATCH ()-[r]->()
            WITH keys(r) AS keys, type(r) AS rel_type
            WITH rel_type, collect(DISTINCT key) AS props
            RETURN {type: rel_type, properties: props} AS output
        """
        rel_props = self._ro_query_database(rel_props_query)

        # Get relationships
        rels_query = """
            MATCH (n)-[r]->(m)
            UNWIND labels(n) AS src_label
            UNWIND labels(m) AS dst_label
            RETURN DISTINCT {
                start: src_label,
                type: type(r),
                end: dst_label
            } AS output
        """
        relationships = self._ro_query_database(rels_query)

        return {
            "node_props": node_props,
            "rel_props": rel_props,
            "relationships": relationships
        }

    def list_graphs(self) -> List[str]:
        """
        List all graphs in the FalkorDB instance.

        Returns:
            List of graph names
        """
        try:
            # The list method may not be available in all FalkorDB versions
            if hasattr(self.db, 'list'):
                result = self.db.list()  # type: ignore
                return result if result else []
            return []
        except Exception:
            return []

    def switch_graph(self, graph_name: str) -> None:
        """
        Switch to a different graph (multi-tenancy).

        Args:
            graph_name: Name of the graph to switch to
        """
        self._graph_name = graph_name
        self.graph = self.db.select_graph(graph_name)
        self.logger.info(f"Switched to graph '{graph_name}'")

    def delete_current_graph(self) -> None:
        """
        Delete the current graph entirely.

        WARNING: This is irreversible.
        """
        self.graph.delete()
        self.logger.info(f"Deleted graph '{self._graph_name}'")

    # =========================================================================
    # Future Extension Points (Vector, FTS, Documents)
    # =========================================================================

    def create_vector_index(
        self,
        label: str,
        attribute: str,
        dimension: int,
        similarity: str = "cosine"
    ) -> None:
        """
        Create a vector index for similarity search.

        Note: FalkorDB 4.0+ supports native vector indexes.

        Args:
            label: Node label to index
            attribute: Property name containing the vector
            dimension: Vector dimension (e.g., 1024 for mistral-embed)
            similarity: Similarity function ('cosine' or 'euclidean')
        """
        # FalkorDB uses positional args for label and property name
        self.graph.create_node_vector_index(
            label,
            attribute,
            dim=dimension,
            similarity_function=similarity
        )
        self.logger.info(f"Created vector index on {label}.{attribute}")

    def vector_search(
        self,
        label: str,
        attribute: str,
        query_vector: List[float],
        k: int = 5
    ) -> List[dict]:
        """
        Perform vector similarity search.

        Args:
            label: Node label to search
            attribute: Vector property name
            query_vector: Query embedding vector
            k: Number of results to return

        Returns:
            List of matching nodes with scores
        """
        query = """
            CALL db.idx.vector.queryNodes(
                $label, $attribute, $k, vecf32($embedding)
            )
            YIELD node, score
            RETURN node, score
        """
        result = self.graph.query(query, {
            'label': label,
            'attribute': attribute,
            'k': k,
            'embedding': query_vector
        })
        return result.result_set if result.result_set else []

    def create_fulltext_index(
        self,
        label: str,
        attributes: List[str]
    ) -> None:
        """
        Create a full-text search index.

        Args:
            label: Node label to index
            attributes: List of property names to index for full-text search
        """
        attrs_str = ", ".join([f"'{a}'" for a in attributes])
        query = f"CALL db.idx.fulltext.createNodeIndex('{label}', {attrs_str})"
        self.graph.query(query)
        self.logger.info(f"Created fulltext index on {label}.{attributes}")

    def fulltext_search(
        self,
        label: str,
        query_text: str,
        k: int = 5
    ) -> List[dict]:
        """
        Perform full-text search.

        Args:
            label: Node label to search
            query_text: Search query
            k: Number of results

        Returns:
            List of matching nodes with scores
        """
        query = """
            CALL db.idx.fulltext.queryNodes($label, $query)
            YIELD node, score
            RETURN node, score
            LIMIT $k
        """
        result = self.graph.query(query, {
            'label': label,
            'query': query_text,
            'k': k
        })
        return result.result_set if result.result_set else []

    # =========================================================================
    # Auto-Index Management
    # =========================================================================

    def create_range_index(
        self,
        label: str,
        attribute: str
    ) -> bool:
        """
        Create a range index for exact match and range queries.

        FalkorDB syntax: CREATE INDEX FOR (n:Label) ON (n.property)

        Args:
            label: Node label to index
            attribute: Property name to index

        Returns:
            True if successful, False otherwise
        """
        try:
            query = f"CREATE INDEX FOR (n:{label}) ON (n.{attribute})"
            self.graph.query(query)
            self.logger.info(f"Created range index on {label}.{attribute}")
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate" in error_str:
                self.logger.debug(f"Range index on {label}.{attribute} already exists")
                return True
            self.logger.warning(f"Failed to create range index on {label}.{attribute}: {e}")
            return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all indexes in the graph.

        Returns:
            List of index information dictionaries with:
            - index_name: Name of the index
            - index_type: Type (RANGE, FULLTEXT, VECTOR)
            - entity_type: 'node' or 'relationship'
            - labels: Labels indexed
            - properties: Properties indexed
            - options: Additional options (dimension, similarity for vector)
        """
        try:
            result = self.graph.query("CALL db.indexes()")
            indexes = []
            for row in result.result_set:
                if isinstance(row, list):
                    # FalkorDB returns: [index_name, index_type, entity_type, labels, properties, options]
                    idx_info = {
                        "index_name": row[0] if len(row) > 0 else None,
                        "index_type": row[1] if len(row) > 1 else None,
                        "entity_type": row[2] if len(row) > 2 else None,
                        "labels": row[3] if len(row) > 3 else [],
                        "properties": row[4] if len(row) > 4 else [],
                        "options": row[5] if len(row) > 5 else {}
                    }
                    indexes.append(idx_info)
            return indexes
        except Exception as e:
            self.logger.warning(f"Failed to list indexes: {e}")
            return []

    def ensure_indexes(
        self,
        labels: Optional[List[str]] = None,
        create_vector: bool = True,
        create_fulltext: bool = True,
        create_range: bool = True
    ) -> Dict[str, bool]:
        """
        Auto-create required indexes for optimal query performance.

        Creates three types of indexes:
        1. **Vector Index** - For semantic similarity search (requires embeddings)
        2. **Fulltext Index** - For text search on content properties
        3. **Range Index** - For exact match lookups on 'id' property

        Args:
            labels: List of labels to index (default: [node_label])
            create_vector: Create vector index if embeddings enabled
            create_fulltext: Create fulltext index for text search
            create_range: Create range index for exact match on 'id'

        Returns:
            Dict mapping index name to success status
        """
        results = {}
        labels = labels or [self._node_label]

        # 1. Create Vector Index for embeddings (semantic search)
        if create_vector and self._embeddings is not None:
            try:
                # Use Cypher syntax: CREATE VECTOR INDEX FOR (n:Label) ON (n.property) OPTIONS {...}
                vector_query = f"""CREATE VECTOR INDEX FOR (n:{self._node_label}) ON (n.{self._embedding_property})
                    OPTIONS {{dimension:{self.MISTRAL_EMBEDDING_DIMENSION}, similarityFunction:'cosine'}}"""
                self.graph.query(vector_query)
                results[f"vector_{self._node_label}_{self._embedding_property}"] = True
                self.logger.info(f"Created vector index on {self._node_label}.{self._embedding_property} (dim={self.MISTRAL_EMBEDDING_DIMENSION})")
            except Exception as e:
                error_str = str(e).lower()
                if "already exists" in error_str or "duplicate" in error_str:
                    results[f"vector_{self._node_label}_{self._embedding_property}"] = True
                    self.logger.debug(f"Vector index already exists on {self._node_label}.{self._embedding_property}")
                else:
                    results[f"vector_{self._node_label}_{self._embedding_property}"] = False
                    self.logger.warning(f"Failed to create vector index: {e}")

        # 2. Create Fulltext Index for content property
        if create_fulltext:
            for label in labels:
                try:
                    # FalkorDB syntax: CALL db.idx.fulltext.createNodeIndex('Label', 'property')
                    fulltext_query = f"CALL db.idx.fulltext.createNodeIndex('{label}', '{self._content_property}')"
                    self.graph.query(fulltext_query)
                    results[f"fulltext_{label}_{self._content_property}"] = True
                    self.logger.info(f"Created fulltext index on {label}.{self._content_property}")
                except Exception as e:
                    error_str = str(e).lower()
                    if "already exists" in error_str or "duplicate" in error_str:
                        results[f"fulltext_{label}_{self._content_property}"] = True
                        self.logger.debug(f"Fulltext index already exists on {label}.{self._content_property}")
                    else:
                        results[f"fulltext_{label}_{self._content_property}"] = False
                        self.logger.warning(f"Failed to create fulltext index on {label}.{self._content_property}: {e}")

        # 3. Create Range Index on 'id' property for fast exact lookups
        if create_range:
            for label in labels:
                try:
                    # FalkorDB syntax: CREATE INDEX FOR (n:Label) ON (n.id)
                    range_query = f"CREATE INDEX FOR (n:{label}) ON (n.id)"
                    self.graph.query(range_query)
                    results[f"range_{label}_id"] = True
                    self.logger.info(f"Created range index on {label}.id")
                except Exception as e:
                    error_str = str(e).lower()
                    if "already exists" in error_str or "duplicate" in error_str:
                        results[f"range_{label}_id"] = True
                        self.logger.debug(f"Range index already exists on {label}.id")
                    else:
                        results[f"range_{label}_id"] = False
                        self.logger.warning(f"Failed to create range index on {label}.id: {e}")

        # Summary logging
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        self.logger.info(f"Index creation complete: {successful}/{total} successful")

        self._indexes_created = True
        return results

    def drop_index(self, index_name: str) -> bool:
        """
        Drop an index by name.

        Args:
            index_name: Name of the index to drop

        Returns:
            True if successful, False otherwise
        """
        try:
            query = f"DROP INDEX {index_name}"
            self.graph.query(query)
            self.logger.info(f"Dropped index {index_name}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to drop index {index_name}: {e}")
            return False