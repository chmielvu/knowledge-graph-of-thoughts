"""Stable JSON output, errors, and agent metadata for the KGoT CLI."""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, NoReturn

import click
import typer

SCHEMA_VERSION = "2.0"
FALLBACK_VERSION = "2.0.0"
EXIT_SUCCESS = 0
EXIT_GENERAL = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_AUTH = 4
EXIT_CONFLICT = 5
EXIT_RATE_LIMITED = 6
EXIT_VALIDATION = 7
EXIT_NETWORK = 8
EXIT_SCHEMA = 9

_FIELD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class CliError(click.ClickException):
    """Expected failure that can be rendered as a stable agent error."""

    def __init__(
        self,
        kind: str,
        code: str,
        message_text: str,
        hint: str,
        context: dict[str, Any] | None = None,
        exit_code: int = EXIT_GENERAL,
    ) -> None:
        super().__init__(message_text)
        self.kind = kind
        self.code = code
        self.message_text = message_text
        self.hint = hint
        self.context = context or {}
        self.cli_exit_code = exit_code
        # Click reads ClickException.exit_code for the process exit code. It is a
        # ClassVar on the base class, so set the instance slot directly.
        self.__dict__["exit_code"] = exit_code

    def show(self, file: Any | None = None) -> None:
        write_json(error_payload(self), sys.stderr if file is None else file)


def package_version() -> str:
    try:
        return metadata.version("kgot")
    except metadata.PackageNotFoundError:
        return FALLBACK_VERSION


def fallback_brief() -> str:
    return (
        "KGoT is a knowledge-graph-backed LLM assistant. Its agent-friendly CLI "
        "provides offline discovery/configuration commands and a validated single "
        "task execution command. The graph backend is fixed to FalkorDB and is not "
        "configurable. JSON is the default output; use --human for prose."
    )


def agent_root() -> Path | None:
    candidates = (
        Path(__file__).resolve().parent / "agent",
        Path(__file__).resolve().parents[1] / "agent",
        Path.cwd() / "agent",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def read_agent_file(relative_path: str, fallback: str = "") -> str:
    root = agent_root()
    if root is None:
        return fallback
    path = root / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return fallback


def _frontmatter(text: str) -> tuple[str, str]:
    name = ""
    description = ""
    if text.startswith("---"):
        lines = text.splitlines()
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"name", "description"}:
                if key.strip() == "name":
                    name = value.strip().strip("\"'")
                else:
                    description = value.strip().strip("\"'")
    return name, description


def agent_rules() -> list[dict[str, str]]:
    defaults = {
        "trigger.md": "Use this CLI for validated KGoT discovery or single-task execution.",
        "workflow.md": "Discover, dry-run, then execute without interactive prompts.",
        "writeback.md": "Use the local issue command to report bugs and bad output.",
    }
    root = agent_root()
    if root is None:
        return [
            {
                "name": name.removesuffix(".md"),
                "description": description,
                "content": description,
            }
            for name, description in defaults.items()
        ]
    result: list[dict[str, str]] = []
    rules_dir = root / "rules"
    for path in sorted(rules_dir.glob("*.md")) if rules_dir.is_dir() else []:
        content = read_agent_file(f"rules/{path.name}", defaults.get(path.name, ""))
        name, description = _frontmatter(content)
        result.append(
            {
                "name": name or path.stem,
                "description": description or defaults.get(path.name, ""),
                "content": content,
            }
        )
    return result


def agent_skills() -> list[dict[str, str]]:
    root = agent_root()
    if root is None:
        return [
            {
                "name": "getting-started",
                "description": "Discover the CLI and run a safe dry-run.",
                "command": "kgot skills show getting-started",
                "content": "Use `kgot --help`, `kgot schema`, then `kgot single --dry-run`.",
            }
        ]
    result: list[dict[str, str]] = []
    skills_dir = root / "skills"
    for path in sorted(skills_dir.glob("*.md")) if skills_dir.is_dir() else []:
        content = read_agent_file(f"skills/{path.name}")
        name, description = _frontmatter(content)
        result.append(
            {
                "name": name or path.stem,
                "description": description,
                "command": f"kgot skills show {name or path.stem}",
                "content": content,
            }
        )
    return result


ISSUE_GUIDE = {
    "command": "kgot issue create --type bug --message 'describe the problem'",
    "types": ["bug", "requirement", "suggestion", "bad-output"],
    "statuses": ["open", "in-progress", "resolved", "closed"],
    "storage": ".kgot/issues.json",
}


def response_metadata() -> dict[str, Any]:
    return {
        "rules": agent_rules(),
        "skills": [
            {
                "name": skill["name"],
                "description": skill["description"],
                "command": skill["command"],
            }
            for skill in agent_skills()
        ],
        "issue": ISSUE_GUIDE,
    }


def success_payload(command: str, result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "version": package_version(),
        "ok": True,
        "command": command,
        "result": result,
        **response_metadata(),
    }
    if isinstance(result, dict):
        for key in ("help", "commands"):
            if key in result:
                payload[key] = result[key]
    return payload


def error_payload(error: CliError) -> dict[str, Any]:
    error_body = {
        "kind": error.kind,
        "code": error.code,
        "message": error.message_text,
        "hint": error.hint,
        "suggestion": error.hint,
        "context": error.context,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "version": package_version(),
        "ok": False,
        "error": error_body,
        "code": error.code,
        "message": error.message_text,
        "suggestion": error.hint,
        **response_metadata(),
    }


def write_json(value: Any, stream: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _parse_fields(fields: str | None) -> list[str]:
    if not fields:
        return []
    names = [item.strip() for item in fields.split(",") if item.strip()]
    invalid = [name for name in names if not _FIELD_PATTERN.fullmatch(name)]
    if invalid:
        raise CliError(
            "validation_error",
            "VALIDATION_ERROR",
            "Invalid field projection.",
            "Use comma-separated JSON field names such as `--fields solution,iterations`.",
            {"fields": invalid},
            EXIT_VALIDATION,
        )
    return names


def _project_result(result: Any, fields: str | None) -> Any:
    names = _parse_fields(fields)
    if not names:
        return result
    if not isinstance(result, dict):
        raise CliError(
            "validation_error",
            "VALIDATION_ERROR",
            "--fields can only project object results.",
            "Remove --fields or request a command that returns an object.",
            {"fields": names},
            EXIT_VALIDATION,
        )
    return {name: result[name] for name in names if name in result}


def _write_human(result: Any) -> None:
    if isinstance(result, str):
        sys.stdout.write(result + ("\n" if not result.endswith("\n") else ""))
    else:
        write_json(result, sys.stdout)


def _write_quiet(result: Any) -> None:
    values: list[Any]
    if isinstance(result, list):
        values = result
    elif isinstance(result, dict) and isinstance(result.get("items"), list):
        values = result["items"]
    else:
        values = [result]
    for value in values:
        if isinstance(value, (dict, list)):
            sys.stdout.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            )
        else:
            sys.stdout.write(f"{value}\n")


def emit_success(
    command: str,
    result: Any,
    *,
    human: bool = False,
    quiet: bool = False,
    fields: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
) -> None:
    _ = dry_run, yes
    projected = _project_result(result, fields)
    if quiet:
        _write_quiet(projected)
    elif human:
        _write_human(projected)
    else:
        write_json(success_payload(command, projected), sys.stdout)
    raise typer.Exit(code=EXIT_SUCCESS)


def fail(
    kind: str,
    code: str,
    message: str,
    hint: str,
    context: dict[str, Any] | None = None,
    exit_code: int = EXIT_GENERAL,
) -> NoReturn:
    raise CliError(kind, code, message, hint, context, exit_code)


def emit_error(error: CliError) -> None:
    write_json(error_payload(error), sys.stderr)
    raise typer.Exit(code=error.cli_exit_code)
