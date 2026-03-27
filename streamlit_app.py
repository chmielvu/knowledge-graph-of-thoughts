"""
KGoT Console - Streamlit UI with Graph Visualization and Tool Transparency

Enhanced UI features:
- Interactive graph visualization with streamlit-flow
- Tool execution transparency with st.status()
- Real-time trace display with categorized stages
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None

try:
    from falkordb import FalkorDB
except Exception:  # pragma: no cover
    FalkorDB = None

# Optional graph visualization - check if package is available
try:
    import streamlit_flow  # noqa: F401
    HAS_STREAMLIT_FLOW = True
except ImportError:
    HAS_STREAMLIT_FLOW = False


st.set_page_config(
    page_title="KGoT Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
KGOT_DIR = ROOT / "kgot"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"


def _read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_env_file(path: Path) -> dict[str, str]:
    env = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


LLM_CONFIG = _read_json(KGOT_DIR / "config_llms.json")
TOOL_CONFIG = _read_json(KGOT_DIR / "config_tools.json")
ENV_CONFIG = _read_env_file(KGOT_DIR / ".env")


# =============================================================================
# Stage Classification for Trace Display
# =============================================================================

STAGE_CONFIG = {
    "planner": {"icon": "🎯", "color": "blue", "label": "Planner"},
    "tools": {"icon": "🔧", "color": "orange", "label": "Tool Call"},
    "graph": {"icon": "🕸️", "color": "green", "label": "Graph Update"},
    "result": {"icon": "✅", "color": "green", "label": "Result"},
    "error": {"icon": "❌", "color": "red", "label": "Error"},
    "log": {"icon": "📋", "color": "gray", "label": "Log"},
    "status": {"icon": "⚡", "color": "blue", "label": "Status"},
}


def _classify_line(line: str) -> str:
    """Classify a log line into a stage type."""
    lowered = line.lower()
    if line.startswith("returned next step") or line.startswith("Reason to insert"):
        return "planner"
    if line.startswith("Tool_calls") or "invoking tool" in lowered:
        return "tools"
    if line.startswith("All nodes and relationships") or line.startswith("Current iteration"):
        return "graph"
    if line.startswith("Solution:"):
        return "result"
    if "traceback" in lowered or "error" in lowered or "critical" in lowered:
        return "error"
    return "log"


def _parse_tool_call(line: str) -> dict | None:
    """Parse tool call information from a log line."""
    if "Tool_calls" in line:
        # Try to extract tool name and args
        match = re.search(r"Tool_calls:\s*(\w+)", line)
        if match:
            return {"tool": match.group(1), "line": line}
    return None


# =============================================================================
# State Management
# =============================================================================

def _init_state():
    """Initialize session state with defaults."""
    st.session_state.setdefault("trace", [])
    st.session_state.setdefault("raw_output", "")
    st.session_state.setdefault("final_answer", "")
    st.session_state.setdefault("last_command", "")
    st.session_state.setdefault(
        "graph_snapshot",
        {
            "backend": "falkordb",
            "nodes": 0,
            "relationships": 0,
            "labels": [],
            "summary": "No graph loaded.",
        },
    )
    st.session_state.setdefault("graph_elements", [])
    st.session_state.setdefault("tool_calls", [])


def _save_uploads(files) -> list[str]:
    if not files:
        return []
    upload_dir = Path(tempfile.mkdtemp(prefix="kgot-streamlit-", dir=str(ROOT)))
    saved_paths = []
    for uploaded in files:
        target = upload_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved_paths.append(str(target))
    return saved_paths


# =============================================================================
# Graph Queries
# =============================================================================

def _query_graph_snapshot(backend: str) -> dict[str, object]:
    """Query graph database for snapshot information."""
    # FalkorDB support
    if backend == "falkordb":
        if FalkorDB is None:
            return {
                "backend": backend,
                "nodes": 0,
                "relationships": 0,
                "labels": [],
                "summary": "FalkorDB package not installed.",
            }
        host = ENV_CONFIG.get("FALKORDB_HOST", "localhost")
        port = int(ENV_CONFIG.get("FALKORDB_PORT", "6379"))
        graph_name = ENV_CONFIG.get("FALKORDB_GRAPH_NAME", "kgot_graph")
        try:
            db = FalkorDB(host=host, port=port)
            graph = db.select_graph(graph_name)

            # Query node and relationship counts
            node_result = graph.query("MATCH (n) RETURN count(n) AS count")
            rel_result = graph.query("MATCH ()-[r]->() RETURN count(r) AS count")
            labels_result = graph.query("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label LIMIT 12")

            # Query actual nodes and edges for visualization
            nodes_result = graph.query("MATCH (n) RETURN n, labels(n) AS labels, id(n) AS id LIMIT 50")
            edges_result = graph.query("MATCH (a)-[r]->(b) RETURN id(a) AS source, id(b) AS target, type(r) AS type LIMIT 100")

            node_count = node_result.result_set[0][0] if node_result.result_set else 0
            rel_count = rel_result.result_set[0][0] if rel_result.result_set else 0
            labels = [row[0] for row in labels_result.result_set] if labels_result.result_set else []

            # Build graph elements for visualization
            elements = []
            node_id_map = {}

            if nodes_result.result_set:
                for i, row in enumerate(nodes_result.result_set):
                    try:
                        node_data = row[0] if row[0] else {}
                        node_labels = row[1] if len(row) > 1 else []
                        node_id = str(row[2]) if len(row) > 2 else f"n{i}"
                        node_id_map[node_id] = f"node_{i}"

                        label = node_labels[0] if node_labels else "Node"
                        name = node_data.get("name", node_data.get("content", "")[:30]) if isinstance(node_data, dict) else str(node_data)[:30]

                        elements.append({
                            "id": f"node_{i}",
                            "label": label,
                            "content": name,
                            "data": node_data,
                        })
                    except Exception:
                        continue

            if edges_result.result_set:
                for row in edges_result.result_set:
                    try:
                        source_id = str(row[0])
                        target_id = str(row[1])
                        rel_type = row[2] if len(row) > 2 else "RELATED"

                        source_idx = node_id_map.get(source_id, source_id)
                        target_idx = node_id_map.get(target_id, target_id)

                        elements.append({
                            "source": source_idx,
                            "target": target_idx,
                            "label": rel_type,
                        })
                    except Exception:
                        continue

            st.session_state.graph_elements = elements

            return {
                "backend": backend,
                "nodes": node_count,
                "relationships": rel_count,
                "labels": labels,
                "summary": f"Live summary from FalkorDB graph '{graph_name}'.",
            }
        except Exception as exc:
            return {
                "backend": backend,
                "nodes": 0,
                "relationships": 0,
                "labels": [],
                "summary": f"Failed to query FalkorDB: {exc}",
            }

    # Neo4j support
    if backend != "neo4j" or GraphDatabase is None:
        return {
            "backend": backend,
            "nodes": 0,
            "relationships": 0,
            "labels": [],
            "summary": f"No live graph summary implemented for backend '{backend}'.",
        }

    uri = ENV_CONFIG.get("NEO4J_URI", "bolt://localhost:7687")
    user = ENV_CONFIG.get("NEO4J_USER", "neo4j")
    password = ENV_CONFIG.get("NEO4J_PASSWORD", "password")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            node_result = session.run("MATCH (n) RETURN count(n) AS count").single()
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
            node_count = node_result["count"] if node_result else 0
            rel_count = rel_result["count"] if rel_result else 0
            labels = [
                record["label"]
                for record in session.run(
                    "MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label LIMIT 12"
                )
            ]
        driver.close()
        return {
            "backend": backend,
            "nodes": node_count,
            "relationships": rel_count,
            "labels": labels,
            "summary": "Live summary loaded from Neo4j.",
        }
    except Exception as exc:
        return {
            "backend": backend,
            "nodes": 0,
            "relationships": 0,
            "labels": [],
            "summary": f"Failed to query Neo4j: {exc}",
        }


# =============================================================================
# KGoT Execution
# =============================================================================

def _run_kgot(prompt: str, backend: str, controller: str, tool_choice: str,
              llm_plan: str, llm_exec: str, uploads, live_placeholder, status_container):
    """Execute KGoT and stream output."""
    file_paths = _save_uploads(uploads)
    command = [
        str(PYTHON_EXE),
        "-m",
        "kgot",
        "--db_choice",
        backend,
        "--controller_choice",
        controller,
        "--tool_choice",
        tool_choice,
        "--llm-plan",
        llm_plan,
        "--llm-exec",
        llm_exec,
        "single",
        "-p",
        prompt,
    ]
    if file_paths:
        command.extend(["--files", *file_paths])

    env = os.environ.copy()
    env.update(ENV_CONFIG)
    env["PYTHONUNBUFFERED"] = "1"

    st.session_state.trace = []
    st.session_state.raw_output = ""
    st.session_state.final_answer = ""
    st.session_state.last_command = " ".join(command)
    st.session_state.tool_calls = []

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    lines: list[str] = []
    current_tool = None

    with status_container:
        tool_status = st.empty()

        if process.stdout is None:
            st.error("Failed to capture process output")
            return 1

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            lines.append(line)

            stage = _classify_line(line)
            st.session_state.trace.append({"stage": stage, "message": line})

            # Track tool calls
            tool_info = _parse_tool_call(line)
            if tool_info:
                current_tool = tool_info["tool"]
                st.session_state.tool_calls.append(tool_info)
                tool_status.info(f"🔧 **Executing tool:** `{current_tool}`")

            if line.startswith("Solution:"):
                st.session_state.final_answer = line.split("Solution:", 1)[1].strip()
                tool_status.success("✅ Execution complete!")

            live_placeholder.code("\n".join(lines[-80:]) or "Waiting for output...")

    return_code = process.wait()
    st.session_state.raw_output = "\n".join(lines)
    if return_code != 0 and not st.session_state.final_answer:
        st.session_state.final_answer = f"Run failed with exit code {return_code}"
    st.session_state.graph_snapshot = _query_graph_snapshot(backend)
    return return_code


# =============================================================================
# UI Components
# =============================================================================

def _render_graph_visualization():
    """Render interactive graph visualization."""
    elements = st.session_state.graph_elements

    if not elements:
        st.info("🕸️ No graph data available. Run a query to populate the knowledge graph.")
        return

    # Separate nodes and edges
    nodes = [e for e in elements if "id" in e]
    edges = [e for e in elements if "source" in e]

    if HAS_STREAMLIT_FLOW and nodes:
        # Use streamlit-flow for interactive visualization
        # These imports are guaranteed available when HAS_STREAMLIT_FLOW is True
        from streamlit_flow import streamlit_flow as _streamlit_flow
        from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
        from streamlit_flow.state import StreamlitFlowState
        from streamlit_flow.layouts import ForceLayout

        st.write(f"📊 **{len(nodes)} nodes, {len(edges)} relationships**")

        try:
            # Convert to streamlit-flow format
            flow_nodes = []
            for i, node in enumerate(nodes[:30]):  # Limit to 30 nodes
                flow_nodes.append(StreamlitFlowNode(
                    node["id"],
                    (i % 6 * 150, i // 6 * 100),
                    {'content': f"**{node['label']}**\n{node.get('content', '')[:50]}"},
                    'default',
                    'bottom',
                    'top'
                ))

            flow_edges = []
            for i, edge in enumerate(edges[:50]):  # Limit edges
                flow_edges.append(StreamlitFlowEdge(
                    f"edge_{i}",
                    edge["source"],
                    edge["target"],
                    animated=True,
                    label=edge.get("label", "")
                ))

            if flow_nodes:
                flow_state = StreamlitFlowState(flow_nodes, flow_edges)

                _streamlit_flow(
                    key='kg_graph',
                    state=flow_state,
                    layout=ForceLayout(node_spacing=100),
                    height=400,
                    fit_view=True,
                    show_controls=True,
                    allow_new_edges=False,
                    get_node_on_click=True,
                    style={'border': '1px solid #ddd', 'borderRadius': '8px'}
                )
        except Exception as e:
            st.warning(f"Graph visualization error: {e}")
            st.json({"nodes": len(nodes), "edges": len(edges)})
    else:
        # Fallback to simple display
        st.write(f"📊 **{len(nodes)} nodes, {len(edges)} relationships**")

        with st.expander("View Nodes", expanded=False):
            for node in nodes[:20]:
                st.markdown(f"- **{node['label']}**: {node.get('content', '')[:60]}")

        with st.expander("View Relationships", expanded=False):
            for edge in edges[:20]:
                st.markdown(f"- `{edge['source']}` → `{edge['target']}` ({edge.get('label', 'RELATED')})")


def _render_execution_trace():
    """Render the execution trace with categorized stages."""
    trace = st.session_state.trace

    if not trace:
        st.info("📋 No execution trace yet. Run a query to see the trace.")
        return

    # Group by stage
    stages = {}
    for item in trace:
        stage = item["stage"]
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(item)

    # Display stage summary
    cols = st.columns(len(STAGE_CONFIG))
    for i, (stage, config) in enumerate(STAGE_CONFIG.items()):
        count = len(stages.get(stage, []))
        if count > 0:
            with cols[i]:
                st.metric(config["label"], count, delta=None)

    # Detailed trace
    st.divider()
    for idx, item in enumerate(trace[-50:], start=max(1, len(trace) - 49)):  # Last 50 items
        config = STAGE_CONFIG.get(item["stage"], STAGE_CONFIG["log"])
        with st.expander(
            f"{config['icon']} {idx}. {config['label']}",
            expanded=item["stage"] in ["tools", "result", "error"]
        ):
            st.code(item["message"], language=None)


# =============================================================================
# Main App
# =============================================================================

_init_state()

# Header
st.title("🧠 Knowledge Graph of Thoughts")
st.caption("Interactive agent with dynamic knowledge graph construction")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    prompt = st.text_area(
        "Prompt",
        value="What is 2+2?",
        height=120,
        help="Enter your question or task for the KGoT agent"
    )

    st.subheader("Model Selection")
    llm_options = list(LLM_CONFIG.keys()) if LLM_CONFIG else ["nanogpt"]
    col1, col2 = st.columns(2)
    with col1:
        llm_plan = st.selectbox("Planning LLM", options=llm_options, index=0) or "nanogpt"
    with col2:
        llm_exec = st.selectbox("Execution LLM", options=llm_options, index=0) or "nanogpt"

    st.subheader("Backend & Controller")
    col1, col2 = st.columns(2)
    with col1:
        backend = st.selectbox("Backend", options=["falkordb", "neo4j", "networkX"], index=0)
    with col2:
        controller = st.selectbox("Controller", options=["queryRetrieve", "directRetrieve"], index=0)

    tool_choice = st.selectbox("Tool bundle", options=["tools_v2_3"], index=0)
    uploads = st.file_uploader("📎 Attachments", accept_multiple_files=True)

    st.divider()
    st.subheader("📊 Runtime Status")
    st.code(
        "\n".join([
            f"Python: {'✅' if PYTHON_EXE.exists() else '❌'} {PYTHON_EXE.name}",
            f"FalkorDB: {ENV_CONFIG.get('FALKORDB_HOST', 'localhost')}:{ENV_CONFIG.get('FALKORDB_PORT', '6379')}",
            f"Neo4j: {ENV_CONFIG.get('NEO4J_URI', 'unset')}",
            f"Attachments: {len(uploads) if uploads else 0}",
        ]),
        language=None
    )

    run_clicked = st.button("🚀 Run", type="primary", use_container_width=True)


# Main content area
left, right = st.columns([1.3, 1.0], gap="medium")

with left:
    # Live trace output
    live_trace = st.empty()
    status_container = st.container()

    if run_clicked:
        # Ensure llm values are strings (selectbox can return None in edge cases)
        llm_plan_val: str = llm_plan or "nanogpt"
        llm_exec_val: str = llm_exec or "nanogpt"
        with st.spinner("Running KGoT agent..."):
            _run_kgot(
                prompt, backend, controller, tool_choice,
                llm_plan_val, llm_exec_val, uploads, live_trace, status_container
            )

    # Execution trace
    st.subheader("📋 Execution Trace")
    _render_execution_trace()

    # Final answer
    if st.session_state.final_answer:
        st.subheader("✅ Final Answer")
        st.success(st.session_state.final_answer)

    # Command display
    with st.expander("🔧 Command", expanded=False):
        st.code(st.session_state.last_command or "No command run yet.", language="bash")

with right:
    # Graph visualization
    st.subheader("🕸️ Knowledge Graph")
    _render_graph_visualization()

    # Graph statistics
    st.divider()
    st.subheader("📈 Graph Statistics")
    snapshot = st.session_state.graph_snapshot

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", snapshot["nodes"])
    with col2:
        st.metric("Relationships", snapshot["relationships"])
    with col3:
        st.metric("Labels", len(snapshot["labels"]))

    if snapshot["labels"]:
        st.write("**Labels:**", ", ".join(f"`{l}`" for l in snapshot["labels"]))

    # Tool calls summary
    if st.session_state.tool_calls:
        st.divider()
        st.subheader("🔧 Tools Used")
        for tc in st.session_state.tool_calls[-10:]:
            st.markdown(f"- `{tc['tool']}`")

    # Raw output (collapsible)
    with st.expander("📄 Raw Output", expanded=False):
        st.code(st.session_state.raw_output[-4000:] if st.session_state.raw_output else "No output yet.")