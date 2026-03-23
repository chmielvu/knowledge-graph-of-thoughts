# Streamlit UI Sketch

This is a lightweight control surface for the clean KGoT repo.

Current scope:
- prompt entry
- model/backend/controller/tool selectors
- attachment upload widget
- execution trace panel
- final answer panel
- graph status / placeholder graph pane

Immediate next wiring steps:
1. call the repo-local `.venv` interpreter with `python -m kgot ...`
2. stream stdout into `st.session_state.trace`
3. parse the final `Solution:` line into the result panel
4. query Neo4j after each run and replace the placeholder graph snapshot
