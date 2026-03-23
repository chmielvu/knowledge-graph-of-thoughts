from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


st.set_page_config(page_title="KGoT Console", page_icon="KG", layout="wide")


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


def _init_state():
    st.session_state.setdefault("trace", [{"stage": "status", "message": "No run started yet."}])
    st.session_state.setdefault("raw_output", "")
    st.session_state.setdefault("final_answer", "")
    st.session_state.setdefault("last_command", "")
    st.session_state.setdefault(
        "graph_snapshot",
        {
            "backend": "neo4j",
            "nodes": 0,
            "relationships": 0,
            "labels": [],
            "summary": "No graph loaded.",
        },
    )


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


def _classify_line(line: str) -> str:
    lowered = line.lower()
    if line.startswith("returned next step") or line.startswith("Reason to insert"):
        return "planner"
    if line.startswith("Tool_calls"):
        return "tools"
    if line.startswith("All nodes and relationships") or line.startswith("Current iteration"):
        return "graph"
    if line.startswith("Solution:"):
        return "result"
    if "traceback" in lowered or "error" in lowered or "critical" in lowered:
        return "error"
    return "log"


def _query_graph_snapshot(backend: str) -> dict[str, object]:
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
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
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


def _run_kgot(prompt: str, backend: str, controller: str, tool_choice: str, llm_plan: str, llm_exec: str, uploads, live_placeholder):
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
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        lines.append(line)
        st.session_state.trace.append({"stage": _classify_line(line), "message": line})
        if line.startswith("Solution:"):
            st.session_state.final_answer = line.split("Solution:", 1)[1].strip()
        live_placeholder.code("\n".join(lines[-80:]) or "Waiting for output...")

    return_code = process.wait()
    st.session_state.raw_output = "\n".join(lines)
    if return_code != 0 and not st.session_state.final_answer:
        st.session_state.final_answer = f"Run failed with exit code {return_code}"
    st.session_state.graph_snapshot = _query_graph_snapshot(backend)
    return return_code


_init_state()

st.title("Knowledge Graph of Thoughts")
st.caption("Local control surface for the clean repo runner.")

with st.sidebar:
    st.header("Run Settings")
    prompt = st.text_area("Prompt", value="What is 2+2?", height=120)
    llm_options = list(LLM_CONFIG.keys()) or ["nanogpt"]
    llm_plan = st.selectbox("Planning model", options=llm_options, index=0)
    llm_exec = st.selectbox("Execution model", options=llm_options, index=0)
    backend = st.selectbox("Backend", options=["neo4j", "networkX", "rdf4j"], index=0)
    controller = st.selectbox("Controller", options=["queryRetrieve", "directRetrieve"], index=0)
    tool_choice = st.selectbox("Tool bundle", options=["tools_v2_3"], index=0)
    uploads = st.file_uploader("Attachments", accept_multiple_files=True)
    st.markdown("**Runtime status**")
    st.code(
        "\n".join(
            [
                f"Python: {PYTHON_EXE}",
                f"Neo4j: {ENV_CONFIG.get('NEO4J_URI', 'unset')}",
                f"Executor: {ENV_CONFIG.get('PYTHON_EXECUTOR_URI', 'unset')}",
                f"Tool config entries: {len(TOOL_CONFIG) if isinstance(TOOL_CONFIG, list) else len(TOOL_CONFIG or {})}",
                f"Queued attachments: {len(uploads) if uploads else 0}",
            ]
        )
    )
    run_clicked = st.button("Run", type="primary", use_container_width=True)


left, right = st.columns([1.35, 1.0], gap="large")

with left:
    live_trace = st.empty()
    if run_clicked:
        with st.spinner("Running KGoT..."):
            _run_kgot(prompt, backend, controller, tool_choice, llm_plan, llm_exec, uploads, live_trace)

    st.subheader("Execution Trace")
    for idx, item in enumerate(st.session_state.trace, start=1):
        with st.expander(f"{idx}. {item['stage']}", expanded=idx > max(len(st.session_state.trace) - 4, 0)):
            st.write(item["message"])

    st.subheader("Final Answer")
    st.text_area("Result", value=st.session_state.final_answer, height=120)

    st.subheader("Command")
    st.code(st.session_state.last_command or "No command run yet.")

with right:
    st.subheader("Graph Status")
    st.json(st.session_state.graph_snapshot)

    st.subheader("Raw Output")
    st.code(st.session_state.raw_output[-6000:] if st.session_state.raw_output else "No output yet.")
