"""Typer command implementations for the refined KGoT CLI."""

from __future__ import annotations

import contextlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
import typer

from cli.output import (
    EXIT_GENERAL,
    EXIT_NOT_FOUND,
    EXIT_SCHEMA,
    EXIT_VALIDATION,
    CliError,
    agent_skills,
    emit_error,
    emit_success,
)
from cli.runtime import (
    COMMANDS,
    GRAPH_BACKEND,
    command_help,
    controller_kwargs,
    effective_runtime,
    group_options,
    output_from_context,
    runtime_from_context,
    sanitized_config,
    validate_runtime,
)
from cli.validate import (
    ID_PATTERN,
    validate_number,
    validate_opaque,
    validate_path,
    validate_text,
)


def schema_command(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    from cli.output import SCHEMA_VERSION

    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested:
        emit_success("schema help", command_help("schema"), **options)
    emit_success(
        "schema", {"schema_version": SCHEMA_VERSION, "commands": COMMANDS}, **options
    )


def single_command(
    ctx: typer.Context,
    problem: str | None = typer.Option(
        None, "--problem", "-p", help="Problem statement to solve."
    ),
    files: list[str] | None = typer.Option(
        None, "--file", "--files", help="Attachment path; repeat for multiple files."
    ),
    controller_choice: str | None = typer.Option(
        None, "--controller-choice", "--controller_choice", help="Controller strategy."
    ),
    tool_choice: str | None = typer.Option(
        None, "--tool-choice", "--tool_choice", help="Tool configuration."
    ),
    iterations: int | None = typer.Option(
        None, "--iterations", "-i", help="Maximum controller iterations."
    ),
    snapshots: str | None = typer.Option(
        None, "--snapshots", "-s", help="Snapshot directory."
    ),
    config_llm_path: str | None = typer.Option(
        None, "--config-llm-path", "--config_llm_path", help="LLM configuration path."
    ),
    logger_level: int | None = typer.Option(
        None, "--logger-level", "--logger_level", help="Python logging level."
    ),
    logger_file_mode: str | None = typer.Option(
        None, "--logger-file-mode", "--logger_file_mode", help="Log file mode."
    ),
    statistics_file: str | None = typer.Option(
        None, "--statistics-file", help="Usage statistics path."
    ),
    num_next_steps_decision: int | None = typer.Option(
        None,
        "--num-next-steps-decision",
        "--num_next_steps_decision",
        help="Planning branch count.",
    ),
    max_retrieve_query_retry: int | None = typer.Option(
        None,
        "--max-retrieve-query-retry",
        "--max_retrieve_query_retry",
        help="Retrieve retry limit.",
    ),
    max_cypher_fixing_retry: int | None = typer.Option(
        None,
        "--max-cypher-fixing-retry",
        "--max_cypher_fixing_retry",
        help="Cypher retry limit.",
    ),
    max_final_solution_parsing: int | None = typer.Option(
        None,
        "--max-final-solution-parsing",
        "--max_final_solution_parsing",
        help="Final parsing retry limit.",
    ),
    max_tool_retries: int | None = typer.Option(
        None, "--max-tool-retries", "--max_tool_retries", help="Tool retry limit."
    ),
    max_llm_retries: int | None = typer.Option(
        None, "--max-llm-retries", "--max_llm_retries", help="LLM retry limit."
    ),
    llm_plan: str | None = typer.Option(None, "--llm-plan", help="Planning model."),
    llm_plan_temp: float | None = typer.Option(
        None, "--llm-plan-temp", help="Planning temperature."
    ),
    llm_exec: str | None = typer.Option(None, "--llm-exec", help="Execution model."),
    llm_exec_temp: float | None = typer.Option(
        None, "--llm-exec-temp", help="Execution temperature."
    ),
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and preview without execution."
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm a destructive operation."),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Guarantee no interactive prompts."
    ),
) -> None:
    from cli.runtime import classify_exception

    options = output_from_context(
        ctx,
        human=human,
        agent=agent,
        quiet=quiet,
        fields=fields,
        dry_run=dry_run,
        yes=yes,
        non_interactive=non_interactive,
    )
    if help_requested:
        emit_success("single help", command_help("single"), **options)
    problem = validate_text(problem, "problem", max_length=100_000)
    attachment_values = files or []
    attachment_paths = [
        validate_path(value, "files", must_exist=True) for value in attachment_values
    ]
    if attachment_paths:
        parents = {str(path.resolve().parent) for path in attachment_paths}
        if len(parents) != 1:
            raise CliError(
                "validation_error",
                "MULTIPLE_ATTACHMENT_DIRECTORIES",
                "All attachments must be in the same directory.",
                "Move attachments into one directory or run separate requests.",
                {"directories": sorted(parents)},
                EXIT_VALIDATION,
            )

    base_runtime = runtime_from_context(ctx)
    runtime = effective_runtime(
        base_runtime,
        {
            "controller_choice": controller_choice,
            "tool_choice": tool_choice,
            "iterations": iterations,
            "snapshots": snapshots,
            "config_llm_path": config_llm_path,
            "logger_level": logger_level,
            "logger_file_mode": logger_file_mode,
            "statistics_file": statistics_file,
            "num_next_steps_decision": num_next_steps_decision,
            "max_retrieve_query_retry": max_retrieve_query_retry,
            "max_cypher_fixing_retry": max_cypher_fixing_retry,
            "max_final_solution_parsing": max_final_solution_parsing,
            "max_tool_retries": max_tool_retries,
            "max_llm_retries": max_llm_retries,
            "llm_plan": llm_plan,
            "llm_plan_temp": llm_plan_temp,
            "llm_exec": llm_exec,
            "llm_exec_temp": llm_exec_temp,
        },
    )
    runtime = validate_runtime(runtime)
    request = {
        "problem": problem,
        "files": [str(path) for path in attachment_paths],
        "configuration": sanitized_config(runtime),
    }
    if options["dry_run"]:
        emit_success(
            "single",
            {
                "dry_run": True,
                "request": request,
                "warning": "Preview only. Remove --dry-run to execute the controller.",
                "suggested_next": [f"kgot single --problem {json.dumps(problem)}"],
            },
            **options,
        )

    try:
        module = importlib.import_module(
            f"kgot.controller.{GRAPH_BACKEND}.{runtime.controller_choice}"
        )
        controller_class = getattr(module, "Controller")
        with contextlib.redirect_stdout(sys.stderr):
            controller = controller_class(**controller_kwargs(runtime))
            attachment_directory = (
                str(attachment_paths[0].parent) if attachment_paths else ""
            )
            attachment_names = [path.name for path in attachment_paths]
            outcome = controller.run(
                problem=problem,
                attachments_file_path=attachment_directory,
                attachments_file_names=attachment_names,
                snapshot_subdir=runtime.snapshots or "",
            )
            if isinstance(outcome, tuple) and len(outcome) >= 2:
                solution, iterations_taken = outcome[0], outcome[1]
            else:
                solution, iterations_taken = outcome, runtime.iterations
            graph = getattr(controller, "graph", None)
            graph_state = (
                graph.get_current_graph_state()
                if graph is not None and hasattr(graph, "get_current_graph_state")
                else {}
            )
            tool_calls = getattr(controller, "tool_calls_made", [])
        emit_success(
            "single",
            {
                "solution": solution,
                "iterations": iterations_taken,
                "graph_state": graph_state,
                "tool_calls": tool_calls,
            },
            **options,
        )
    except click.exceptions.Exit:
        raise
    except CliError:
        raise
    except Exception as exc:
        emit_error(classify_exception(exc))


def config_callback(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    group_options(ctx, help_requested, human, agent, quiet, fields)


def config_show(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested:
        emit_success(
            "config show help",
            {
                "help": "Print sanitized effective runtime configuration.",
                "command": "kgot config show",
            },
            **options,
        )
    emit_success(
        "config show",
        sanitized_config(validate_runtime(runtime_from_context(ctx))),
        **options,
    )


def tools_callback(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    group_options(ctx, help_requested, human, agent, quiet, fields)


def tools_list(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested:
        emit_success("tools list help", command_help("tools"), **options)
    emit_success(
        "tools list",
        {
            "items": [
                {
                    "name": "tools_v2_3",
                    "description": "Current KGoT tool manager configuration.",
                }
            ],
            "total": 1,
            "limit": 1,
            "offset": 0,
            "has_more": False,
        },
        **options,
    )


def skills_callback(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    group_options(ctx, help_requested, human, agent, quiet, fields)


def skills_list(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested:
        emit_success("skills list help", command_help("skills"), **options)
    emit_success(
        "skills list",
        {
            "items": [
                {key: skill[key] for key in ("name", "description", "command")}
                for skill in agent_skills()
            ]
        },
        **options,
    )


def skills_show(
    ctx: typer.Context,
    name: str | None = typer.Argument(None, help="Skill name."),
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested:
        emit_success("skills show help", command_help("skills"), **options)
    name = validate_opaque(name or "", "name")
    matches = [skill for skill in agent_skills() if skill["name"] == name]
    if not matches:
        raise CliError(
            "not_found",
            "NOT_FOUND",
            f"Unknown skill: {name}.",
            "Run `kgot skills list` to see available skills.",
            {"name": name},
            EXIT_NOT_FOUND,
        )
    emit_success("skills show", matches[0], **options)


# Issue persistence is deliberately local, bounded, and independent of the graph.
def _issue_path() -> Path:
    return Path.cwd() / ".kgot" / "issues.json"


def _read_issues() -> list[dict[str, Any]]:
    path = _issue_path()
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(
            "schema_error",
            "SCHEMA_ERROR",
            "The local issue store is unreadable.",
            "Repair or remove .kgot/issues.json, then retry.",
            {"path": str(path), "exception_type": exc.__class__.__name__},
            EXIT_SCHEMA,
        )
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CliError(
            "schema_error",
            "SCHEMA_ERROR",
            "The local issue store has an invalid schema.",
            "Repair or remove .kgot/issues.json, then retry.",
            {"path": str(path)},
            EXIT_SCHEMA,
        )
    return value


def _write_issues(issues: list[dict[str, Any]]) -> None:
    path = _issue_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(issues, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError as exc:
        raise CliError(
            "storage_error",
            "GENERAL_ERROR",
            "The local issue store could not be written.",
            "Check write permissions for .kgot and retry.",
            {"exception_type": exc.__class__.__name__},
            EXIT_GENERAL,
        )


def _issue_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ISSUE-{timestamp}-{uuid4().hex[:8].upper()}"


def issue_callback(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    group_options(ctx, help_requested, human, agent, quiet, fields)


def issue_create(
    ctx: typer.Context,
    issue_type: str | None = typer.Option(
        None, "--type", help="bug, requirement, suggestion, or bad-output."
    ),
    message: str | None = typer.Option(None, "--message", help="Issue description."),
    issue_context: str | None = typer.Option(
        None, "--context", help="Optional JSON context object."
    ),
    exit_code: int | None = typer.Option(
        None, "--exit-code", help="Exit code observed by the reporter."
    ),
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing."),
) -> None:
    from cli.output import ISSUE_GUIDE, package_version

    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields, dry_run=dry_run
    )
    if help_requested:
        emit_success("issue create help", command_help("issue"), **options)
    issue_type = validate_opaque(issue_type or "", "type")
    if issue_type not in ISSUE_GUIDE["types"]:
        raise CliError(
            "validation_error",
            "INVALID_CHOICE",
            "Unsupported issue type.",
            "Use bug, requirement, suggestion, or bad-output.",
            {"type": issue_type},
            EXIT_VALIDATION,
        )
    message = validate_text(message, "message", max_length=20_000)
    parsed_context: Any = {}
    if issue_context:
        validate_text(issue_context, "context", max_length=20_000)
        try:
            context_value = json.loads(issue_context)
        except json.JSONDecodeError:
            raise CliError(
                "validation_error",
                "INVALID_JSON",
                "--context must be a JSON object.",
                'Pass valid JSON such as `--context \'{"command":"kgot schema"}\'`.',
                {},
                EXIT_VALIDATION,
            )
        if not isinstance(context_value, dict):
            raise CliError(
                "validation_error",
                "INVALID_JSON",
                "--context must be a JSON object.",
                "Pass a JSON object, not an array or scalar.",
                {},
                EXIT_VALIDATION,
            )
        parsed_context = context_value
    if exit_code is not None:
        validate_number(exit_code, "exit_code", 0, 255)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    issue = {
        "id": _issue_id(),
        "type": issue_type,
        "status": "open",
        "message": message,
        "context": parsed_context,
        "version": package_version(),
        "exit_code": exit_code,
        "created_at": now,
        "updated_at": now,
    }
    if options["dry_run"]:
        emit_success(
            "issue create",
            {
                "dry_run": True,
                "issue": issue,
                "warning": "Preview only. Remove --dry-run to write the issue.",
            },
            **options,
        )
    issues = _read_issues()
    issues.append(issue)
    _write_issues(issues)
    emit_success("issue create", {"issue": issue}, **options)


def issue_list(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", help="Maximum issues to return."),
    offset: int = typer.Option(0, "--offset", help="Number of issues to skip."),
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested:
        emit_success("issue list help", command_help("issue"), **options)
    validate_number(limit, "limit", 1, 50)
    validate_number(offset, "offset", 0)
    issues = _read_issues()
    items = issues[offset : offset + limit]
    emit_success(
        "issue list",
        {
            "items": items,
            "total": len(issues),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(issues),
        },
        **options,
    )


def issue_show(
    ctx: typer.Context,
    issue_id: str | None = typer.Argument(None, help="Issue identifier."),
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
) -> None:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested:
        emit_success("issue show help", command_help("issue"), **options)
    issue_id = validate_opaque(issue_id or "", "id")
    if not ID_PATTERN.fullmatch(issue_id):
        raise CliError(
            "validation_error",
            "INVALID_ID",
            "Invalid issue identifier.",
            "Use an ID returned by `kgot issue list`.",
            {"id": issue_id},
            EXIT_VALIDATION,
        )
    issue = next((item for item in _read_issues() if item.get("id") == issue_id), None)
    if issue is None:
        raise CliError(
            "not_found",
            "NOT_FOUND",
            f"Issue does not exist: {issue_id}.",
            "Run `kgot issue list` to see stored issue IDs.",
            {"id": issue_id},
            EXIT_NOT_FOUND,
        )
    emit_success("issue show", {"issue": issue}, **options)


def issue_close(
    ctx: typer.Context,
    issue_id: str | None = typer.Argument(None, help="Issue identifier."),
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured command help."
    ),
    human: bool = typer.Option(False, "--human", help="Use readable output."),
    agent: bool = typer.Option(False, "--agent", help="Force JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Emit bare values."),
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated result fields."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing."),
) -> None:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields, dry_run=dry_run
    )
    if help_requested:
        emit_success("issue close help", command_help("issue"), **options)
    issue_id = validate_opaque(issue_id or "", "id")
    if not ID_PATTERN.fullmatch(issue_id):
        raise CliError(
            "validation_error",
            "INVALID_ID",
            "Invalid issue identifier.",
            "Use an ID returned by `kgot issue list`.",
            {"id": issue_id},
            EXIT_VALIDATION,
        )
    issues = _read_issues()
    issue = next((item for item in issues if item.get("id") == issue_id), None)
    if issue is None:
        raise CliError(
            "not_found",
            "NOT_FOUND",
            f"Issue does not exist: {issue_id}.",
            "Run `kgot issue list` to see stored issue IDs.",
            {"id": issue_id},
            EXIT_NOT_FOUND,
        )
    updated = {
        **issue,
        "status": "closed",
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if options["dry_run"]:
        emit_success(
            "issue close",
            {
                "dry_run": True,
                "issue": updated,
                "warning": "Preview only. Remove --dry-run to write the issue.",
            },
            **options,
        )
    _write_issues(
        [{**item, **updated} if item.get("id") == issue_id else item for item in issues]
    )
    emit_success("issue close", {"issue": updated}, **options)


__all__ = [
    "config_callback",
    "config_show",
    "issue_callback",
    "issue_close",
    "issue_create",
    "issue_list",
    "issue_show",
    "schema_command",
    "single_command",
    "skills_callback",
    "skills_list",
    "skills_show",
    "tools_callback",
    "tools_list",
]
