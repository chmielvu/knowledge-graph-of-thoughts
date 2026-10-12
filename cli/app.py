"""Typer application assembly for the refined KGoT CLI."""

from __future__ import annotations

import os
import sys
from typing import Sequence

import click
import typer
from dotenv import load_dotenv

from cli.commands import (
    config_callback,
    config_show,
    issue_callback,
    issue_close,
    issue_create,
    issue_list,
    issue_show,
    schema_command,
    single_command,
    skills_callback,
    skills_list,
    skills_show,
    tools_callback,
    tools_list,
)
from cli.output import (
    EXIT_SUCCESS,
    EXIT_USAGE,
    SCHEMA_VERSION,
    CliError,
    agent_skills,
    emit_success,
    error_payload,
    fallback_brief,
    package_version,
    read_agent_file,
    write_json,
)
from cli.runtime import COMMANDS, RuntimeConfig, classify_exception, root_help

app = typer.Typer(
    name="kgot",
    help=fallback_brief(),
    no_args_is_help=False,
    invoke_without_command=True,
    add_help_option=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    add_completion=True,
)
skills_app = typer.Typer(
    name="skills",
    no_args_is_help=False,
    invoke_without_command=True,
    add_help_option=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
config_app = typer.Typer(
    name="config",
    no_args_is_help=False,
    invoke_without_command=True,
    add_help_option=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
tools_app = typer.Typer(
    name="tools",
    no_args_is_help=False,
    invoke_without_command=True,
    add_help_option=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
issue_app = typer.Typer(
    name="issue",
    no_args_is_help=False,
    invoke_without_command=True,
    add_help_option=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
app.add_typer(
    config_app,
    name="config",
    help="Inspect effective non-secret runtime configuration.",
)
app.add_typer(
    tools_app, name="tools", help="List supported offline tool configurations."
)
app.add_typer(skills_app, name="skills", help="List or inspect agent skills.")
app.add_typer(issue_app, name="issue", help="Store local structured feedback.")


@app.callback()
def root_callback(
    ctx: typer.Context,
    help_requested: bool = typer.Option(
        False, "--help", help="Emit structured JSON help."
    ),
    brief: bool = typer.Option(
        False, "--brief", help="Emit the one-paragraph agent brief."
    ),
    version: bool = typer.Option(
        False, "--version", "-v", help="Emit the CLI version."
    ),
    schema_requested: bool = typer.Option(
        False, "--schema", help="Emit the command schema."
    ),
    skills_requested: bool = typer.Option(
        False, "--skills", help="List shipped agent skills."
    ),
    human: bool = typer.Option(
        False,
        "--human",
        envvar="KGOT_OUTPUT_HUMAN",
        help="Use readable output instead of JSON.",
    ),
    agent: bool = typer.Option(
        False, "--agent", envvar="KGOT_OUTPUT_AGENT", help="Force JSON agent output."
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        envvar="KGOT_QUIET",
        help="Emit bare values for pipelines.",
    ),
    fields: str | None = typer.Option(
        None, "--fields", envvar="KGOT_FIELDS", help="Comma-separated result fields."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        envvar="KGOT_DRY_RUN",
        help="Preview a mutating or execution command.",
    ),
    yes: bool = typer.Option(
        False, "--yes", envvar="KGOT_YES", help="Confirm a destructive operation."
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        envvar="KGOT_NON_INTERACTIVE",
        help="Guarantee no interactive prompts.",
    ),
    controller_choice: str = typer.Option(
        "queryRetrieve",
        "--controller-choice",
        "--controller_choice",
        envvar="KGOT_CONTROLLER_CHOICE",
        help="Controller strategy.",
    ),
    tool_choice: str = typer.Option(
        "tools_v2_3",
        "--tool-choice",
        "--tool_choice",
        envvar="KGOT_TOOL_CHOICE",
        help="Tool configuration.",
    ),
    iterations: int = typer.Option(
        7,
        "--iterations",
        "-i",
        envvar="KGOT_ITERATIONS",
        help="Maximum controller iterations.",
    ),
    snapshots: str | None = typer.Option(
        None, "--snapshots", "-s", envvar="KGOT_SNAPSHOTS", help="Snapshot directory."
    ),
    config_llm_path: str = typer.Option(
        "kgot/config_llms.json",
        "--config-llm-path",
        "--config_llm_path",
        envvar="KGOT_CONFIG_LLM_PATH",
        help="LLM configuration path.",
    ),
    logger_level: int = typer.Option(
        20,
        "--logger-level",
        "--logger_level",
        envvar="KGOT_LOGGER_LEVEL",
        help="Python logging level.",
    ),
    logger_file_mode: str = typer.Option(
        "a",
        "--logger-file-mode",
        "--logger_file_mode",
        envvar="KGOT_LOGGER_FILE_MODE",
        help="Log file mode.",
    ),
    statistics_file: str = typer.Option(
        "llm_cost.json",
        "--statistics-file",
        envvar="KGOT_STATISTICS_FILE",
        help="Usage statistics path.",
    ),
    num_next_steps_decision: int = typer.Option(
        5,
        "--num-next-steps-decision",
        "--num_next_steps_decision",
        envvar="KGOT_NUM_NEXT_STEPS_DECISION",
        help="Planning branch count.",
    ),
    max_retrieve_query_retry: int = typer.Option(
        3,
        "--max-retrieve-query-retry",
        "--max_retrieve_query_retry",
        envvar="KGOT_MAX_RETRIEVE_QUERY_RETRY",
        help="Retrieve retry limit.",
    ),
    max_cypher_fixing_retry: int = typer.Option(
        3,
        "--max-cypher-fixing-retry",
        "--max_cypher_fixing_retry",
        envvar="KGOT_MAX_CYPHER_FIXING_RETRY",
        help="Cypher retry limit.",
    ),
    max_final_solution_parsing: int = typer.Option(
        3,
        "--max-final-solution-parsing",
        "--max_final_solution_parsing",
        envvar="KGOT_MAX_FINAL_SOLUTION_PARSING",
        help="Final parsing retry limit.",
    ),
    max_tool_retries: int = typer.Option(
        6,
        "--max-tool-retries",
        "--max_tool_retries",
        envvar="KGOT_MAX_TOOL_RETRIES",
        help="Tool retry limit.",
    ),
    max_llm_retries: int = typer.Option(
        6,
        "--max-llm-retries",
        "--max_llm_retries",
        envvar="KGOT_MAX_LLM_RETRIES",
        help="LLM retry limit.",
    ),
    llm_plan: str = typer.Option(
        "gpt-4o-mini", "--llm-plan", envvar="KGOT_LLM_PLAN", help="Planning model."
    ),
    llm_plan_temp: float = typer.Option(
        0.0,
        "--llm-plan-temp",
        envvar="KGOT_LLM_PLAN_TEMP",
        help="Planning temperature.",
    ),
    llm_exec: str = typer.Option(
        "gpt-4o-mini", "--llm-exec", envvar="KGOT_LLM_EXEC", help="Execution model."
    ),
    llm_exec_temp: float = typer.Option(
        0.0,
        "--llm-exec-temp",
        envvar="KGOT_LLM_EXEC_TEMP",
        help="Execution temperature.",
    ),
) -> None:
    """KGoT agent-friendly command line interface."""
    load_dotenv(override=False)
    runtime = RuntimeConfig(
        controller_choice=controller_choice,
        tool_choice=tool_choice,
        iterations=iterations,
        snapshots=snapshots,
        config_llm_path=config_llm_path,
        logger_level=logger_level,
        logger_file_mode=logger_file_mode,
        statistics_file=statistics_file,
        num_next_steps_decision=num_next_steps_decision,
        max_retrieve_query_retry=max_retrieve_query_retry,
        max_cypher_fixing_retry=max_cypher_fixing_retry,
        max_final_solution_parsing=max_final_solution_parsing,
        max_tool_retries=max_tool_retries,
        max_llm_retries=max_llm_retries,
        llm_plan=llm_plan,
        llm_plan_temp=llm_plan_temp,
        llm_exec=llm_exec,
        llm_exec_temp=llm_exec_temp,
        python_executor_uri=os.getenv(
            "PYTHON_EXECUTOR_URI", "http://localhost:16000/run"
        ),
        falkordb_host=os.getenv("FALKORDB_HOST", "localhost"),
        falkordb_port=int(os.getenv("FALKORDB_PORT", "6379")),
        falkordb_password=os.getenv("FALKORDB_PASSWORD"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )
    ctx.ensure_object(dict)
    ctx.obj["runtime"] = runtime
    ctx.obj["output"] = {
        "human": human,
        "agent": agent,
        "quiet": quiet,
        "fields": fields,
        "dry_run": dry_run,
        "yes": yes,
        "non_interactive": non_interactive,
    }

    if brief:
        emit_success(
            "brief",
            {"brief": read_agent_file("brief.md", fallback_brief())},
            human=human and not agent,
            quiet=quiet,
            fields=fields,
        )
    if version:
        emit_success(
            "version",
            {"version": package_version()},
            human=human and not agent,
            quiet=quiet,
            fields=fields,
        )
    if schema_requested:
        emit_success(
            "schema",
            {"schema_version": SCHEMA_VERSION, "commands": COMMANDS},
            human=human and not agent,
            quiet=quiet,
            fields=fields,
        )
    if skills_requested:
        emit_success(
            "skills",
            {
                "items": [
                    {key: skill[key] for key in ("name", "description", "command")}
                    for skill in agent_skills()
                ]
            },
            human=human and not agent,
            quiet=quiet,
            fields=fields,
        )
    if help_requested or ctx.invoked_subcommand is None:
        emit_success(
            "help", root_help(), human=human and not agent, quiet=quiet, fields=fields
        )


app.command("schema")(schema_command)
app.command("single")(single_command)
config_app.callback()(config_callback)
config_app.command("show")(config_show)
tools_app.callback()(tools_callback)
tools_app.command("list")(tools_list)
skills_app.callback()(skills_callback)
skills_app.command("list")(skills_list)
skills_app.command("show")(skills_show)
issue_app.callback()(issue_callback)
issue_app.command("create")(issue_create)
issue_app.command("list")(issue_list)
issue_app.command("show")(issue_show)
issue_app.command("close")(issue_close)


def main(args: Sequence[str] | None = None) -> int:
    """Run the Typer app and normalize parser failures to JSON errors."""
    try:
        app(
            args=list(args) if args is not None else None,
            prog_name="kgot",
            standalone_mode=False,
        )
    except CliError as error:
        write_json(error_payload(error), sys.stderr)
        return error.cli_exit_code
    except click.exceptions.Exit as exit_exception:
        return int(exit_exception.exit_code)
    except click.exceptions.UsageError as error:
        normalized = CliError(
            "usage_error",
            "USAGE_ERROR",
            str(error),
            "Run `kgot --help` or `kgot schema` for valid commands and parameters.",
            {},
            EXIT_USAGE,
        )
        write_json(error_payload(normalized), sys.stderr)
        return EXIT_USAGE
    except Exception as error:
        normalized = classify_exception(error)
        write_json(error_payload(normalized), sys.stderr)
        return normalized.cli_exit_code
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = ["app", "main"]
