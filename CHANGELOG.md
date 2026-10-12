# Changelog

## 2.0.0 — 2026-10-13

Breaking CLI release. The agent-friendly CLI moved from `kgot/cli.py` to the
`cli/` package (`cli/output.py`, `cli/validate.py`, `cli/runtime.py`,
`cli/commands.py`, `cli/app.py`). The `kgot` entry point now runs
`cli.app:main`; `python -m cli` is equivalent and `python -m kgot` delegates to
the same implementation.

Breaking changes:

- The graph backend is fixed to FalkorDB and is no longer exposed:
  `--db-choice` / `--db_choice` and `KGOT_DB_CHOICE` were removed, and
  `db_choice`, `neo4j_uri`, `neo4j_user`, and `neo4j_password_set` were removed
  from `config show`, dry-run previews, and the command schema. Execution always
  uses `kgot.controller.falkordb.<controller_choice>`.
- CLI JSON `schema_version` bumped from `1.0` to `2.0`.

Compatible additions:

- `single --dry-run` includes `suggested_next` with the matching execute command.
- `schema` documents every command parameter, default, environment variable, and
  example.
- `--non-interactive` / `KGOT_NON_INTERACTIVE` is accepted as an explicit
  no-prompt guarantee.

Migration:

- Replace `kgot --db_choice <backend> ... single ...` with `kgot ... single ...`.
- Replace `python -m kgot --db_choice <backend> ...` with `python -m cli ...`.
- The Streamlit console no longer offers a backend selector; it always uses
  FalkorDB.
