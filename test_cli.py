"""Focused tests for the refined agent-friendly KGoT CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

import cli.commands as cli_commands
from cli.app import app


runner = CliRunner()


def read_json(result, stream: str = "stdout") -> dict:
    return json.loads(getattr(result, stream))


def test_root_help_is_structured_json() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    payload = read_json(result)
    assert payload["schema_version"] == "2.0"
    assert payload["ok"] is True
    assert payload["result"]["commands"]
    assert payload["commands"]
    assert payload["rules"]
    assert payload["skills"]
    assert payload["issue"]["types"]


def test_single_dry_run_is_offline_and_structured() -> None:
    result = runner.invoke(app, ["single", "--problem", "What is KGoT?", "--dry-run"])

    assert result.exit_code == 0
    payload = read_json(result)
    assert payload["command"] == "single"
    assert payload["result"]["dry_run"] is True
    assert payload["result"]["request"]["problem"] == "What is KGoT?"
    assert result.stderr == ""


def test_missing_problem_is_structured_stderr_error() -> None:
    result = runner.invoke(app, ["single", "--dry-run"])

    assert result.exit_code == 2
    payload = read_json(result, "stderr")
    assert payload["ok"] is False
    assert payload["error"]["kind"] == "validation_error"
    assert payload["error"]["code"] == "MISSING_REQUIRED"
    assert result.stdout == ""


def test_path_traversal_is_rejected() -> None:
    result = runner.invoke(
        app, ["single", "--problem", "x", "--snapshots", "../outside", "--dry-run"]
    )

    assert result.exit_code == 7
    payload = read_json(result, "stderr")
    assert payload["error"]["code"] == "PATH_TRAVERSAL_REJECTED"


def test_db_choice_flag_is_rejected(capsys) -> None:
    from cli.app import main

    exit_code = main(["single", "--problem", "x", "--db-choice", "neo4j", "--dry-run"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "USAGE_ERROR"


def test_config_show_has_no_backend_keys() -> None:
    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    configuration = read_json(result)["result"]
    assert "db_choice" not in configuration
    assert "neo4j_uri" not in configuration
    assert "neo4j_user" not in configuration
    assert "neo4j_password_set" not in configuration
    assert configuration["falkordb_port"] == 6379


def test_schema_documents_no_backend_params() -> None:
    result = runner.invoke(app, ["schema"])

    assert result.exit_code == 0
    commands = read_json(result)["result"]["commands"]
    single = next(command for command in commands if command["name"] == "single")
    assert all(param["name"] != "db_choice" for param in single["params"])
    assert "db_choice" not in json.dumps(commands)
    assert single["examples"]


def test_single_execution_uses_existing_controller_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeGraph:
        def get_current_graph_state(self):
            return {"nodes": 1}

    class FakeController:
        tool_calls_made = [{"name": "fake"}]
        graph = FakeGraph()

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            print("controller diagnostic")
            captured["run"] = kwargs
            return "answer", 2

    def fake_import(name):
        assert name == "kgot.controller.falkordb.queryRetrieve"
        return SimpleNamespace(Controller=FakeController)

    monkeypatch.setattr(cli_commands.importlib, "import_module", fake_import)
    result = runner.invoke(app, ["single", "--problem", "execute this"])

    assert result.exit_code == 0
    payload = read_json(result)
    assert payload["result"]["solution"] == "answer"
    assert payload["result"]["iterations"] == 2
    assert payload["result"]["graph_state"] == {"nodes": 1}
    assert captured["llm_planning_model"] == "gpt-4o-mini"
    assert captured["db_choice"] == "falkordb"
    assert captured["falkordb_host"] == "localhost"
    assert "neo4j_username" not in captured
    assert "neo4j_pwd" not in captured
    assert "controller diagnostic" in result.stderr


def test_issue_feedback_lifecycle_uses_local_json_store() -> None:
    with runner.isolated_filesystem():
        created = runner.invoke(
            app, ["issue", "create", "--type", "bug", "--message", "bad output"]
        )
        assert created.exit_code == 0
        issue = read_json(created)["result"]["issue"]
        issue_id = issue["id"]

        listed = runner.invoke(app, ["issue", "list"])
        assert listed.exit_code == 0
        assert read_json(listed)["result"]["total"] == 1

        closed = runner.invoke(app, ["issue", "close", issue_id])
        assert closed.exit_code == 0
        assert read_json(closed)["result"]["issue"]["status"] == "closed"

        shown = runner.invoke(app, ["issue", "show", issue_id])
        assert shown.exit_code == 0
        assert read_json(shown)["result"]["issue"]["status"] == "closed"
