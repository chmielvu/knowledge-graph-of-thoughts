"""Runtime configuration for the refined KGoT CLI.

The graph backend is deliberately fixed to FalkorDB and is not exposed as a
flag, environment variable, configuration key, or schema field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import typer

from cli.output import (
    EXIT_AUTH,
    EXIT_GENERAL,
    EXIT_NETWORK,
    EXIT_RATE_LIMITED,
    EXIT_USAGE,
    EXIT_VALIDATION,
    SCHEMA_VERSION,
    CliError,
    agent_rules,
    agent_skills,
    emit_success,
    fail,
    fallback_brief,
    package_version,
    read_agent_file,
)
from cli.validate import validate_choice, validate_number, validate_path

GRAPH_BACKEND = "falkordb"


@dataclass(frozen=True)
class RuntimeConfig:
    """Normalized options used by the existing FalkorDB controllers."""

    controller_choice: str = "queryRetrieve"
    tool_choice: str = "tools_v2_3"
    iterations: int = 7
    snapshots: str | None = None
    config_llm_path: str = "kgot/config_llms.json"
    logger_level: int = 20
    logger_file_mode: str = "a"
    statistics_file: str = "llm_cost.json"
    num_next_steps_decision: int = 5
    max_retrieve_query_retry: int = 3
    max_cypher_fixing_retry: int = 3
    max_final_solution_parsing: int = 3
    max_tool_retries: int = 6
    max_llm_retries: int = 6
    llm_plan: str = "gpt-4o-mini"
    llm_plan_temp: float = 0.0
    llm_exec: str = "gpt-4o-mini"
    llm_exec_temp: float = 0.0
    python_executor_uri: str = "http://localhost:16000/run"
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str | None = None
    mistral_api_key: str | None = None


def _param(
    name: str,
    param_type: str,
    *,
    required: bool,
    flag: str | None = None,
    default: Any = None,
    description: str = "",
    envvar: str | None = None,
) -> dict[str, Any]:
    param: dict[str, Any] = {"name": name, "type": param_type, "required": required}
    if flag is not None:
        param["flag"] = flag
    if default is not None:
        param["default"] = default
    if description:
        param["description"] = description
    if envvar is not None:
        param["envvar"] = envvar
    return param


def _global_runtime_params() -> list[dict[str, Any]]:
    return [
        _param(
            "controller_choice",
            "enum[directRetrieve,queryRetrieve]",
            required=False,
            flag="--controller-choice",
            default="queryRetrieve",
            description="Controller strategy.",
            envvar="KGOT_CONTROLLER_CHOICE",
        ),
        _param(
            "tool_choice",
            "enum[tools_v2_3]",
            required=False,
            flag="--tool-choice",
            default="tools_v2_3",
            description="Tool configuration.",
            envvar="KGOT_TOOL_CHOICE",
        ),
        _param(
            "iterations",
            "integer[1,20]",
            required=False,
            flag="--iterations/-i",
            default=7,
            description="Maximum controller iterations.",
            envvar="KGOT_ITERATIONS",
        ),
        _param(
            "snapshots",
            "string",
            required=False,
            flag="--snapshots/-s",
            description="Snapshot directory.",
            envvar="KGOT_SNAPSHOTS",
        ),
        _param(
            "config_llm_path",
            "string",
            required=False,
            flag="--config-llm-path",
            default="kgot/config_llms.json",
            description="LLM configuration path.",
            envvar="KGOT_CONFIG_LLM_PATH",
        ),
        _param(
            "logger_level",
            "integer",
            required=False,
            flag="--logger-level",
            default=20,
            description="Python logging level.",
            envvar="KGOT_LOGGER_LEVEL",
        ),
        _param(
            "logger_file_mode",
            "enum[a,w,x]",
            required=False,
            flag="--logger-file-mode",
            default="a",
            description="Log file mode.",
            envvar="KGOT_LOGGER_FILE_MODE",
        ),
        _param(
            "statistics_file",
            "string",
            required=False,
            flag="--statistics-file",
            default="llm_cost.json",
            description="Usage statistics path.",
            envvar="KGOT_STATISTICS_FILE",
        ),
        _param(
            "num_next_steps_decision",
            "integer[1,100]",
            required=False,
            flag="--num-next-steps-decision",
            default=5,
            description="Planning branch count.",
            envvar="KGOT_NUM_NEXT_STEPS_DECISION",
        ),
        _param(
            "max_retrieve_query_retry",
            "integer[0,100]",
            required=False,
            flag="--max-retrieve-query-retry",
            default=3,
            description="Retrieve retry limit.",
            envvar="KGOT_MAX_RETRIEVE_QUERY_RETRY",
        ),
        _param(
            "max_cypher_fixing_retry",
            "integer[0,100]",
            required=False,
            flag="--max-cypher-fixing-retry",
            default=3,
            description="Cypher retry limit.",
            envvar="KGOT_MAX_CYPHER_FIXING_RETRY",
        ),
        _param(
            "max_final_solution_parsing",
            "integer[0,100]",
            required=False,
            flag="--max-final-solution-parsing",
            default=3,
            description="Final parsing retry limit.",
            envvar="KGOT_MAX_FINAL_SOLUTION_PARSING",
        ),
        _param(
            "max_tool_retries",
            "integer[0,100]",
            required=False,
            flag="--max-tool-retries",
            default=6,
            description="Tool retry limit.",
            envvar="KGOT_MAX_TOOL_RETRIES",
        ),
        _param(
            "max_llm_retries",
            "integer[0,100]",
            required=False,
            flag="--max-llm-retries",
            default=6,
            description="LLM retry limit.",
            envvar="KGOT_MAX_LLM_RETRIES",
        ),
        _param(
            "llm_plan",
            "string",
            required=False,
            flag="--llm-plan",
            default="gpt-4o-mini",
            description="Planning model.",
            envvar="KGOT_LLM_PLAN",
        ),
        _param(
            "llm_plan_temp",
            "number[0.0,2.0]",
            required=False,
            flag="--llm-plan-temp",
            default=0.0,
            description="Planning temperature.",
            envvar="KGOT_LLM_PLAN_TEMP",
        ),
        _param(
            "llm_exec",
            "string",
            required=False,
            flag="--llm-exec",
            default="gpt-4o-mini",
            description="Execution model.",
            envvar="KGOT_LLM_EXEC",
        ),
        _param(
            "llm_exec_temp",
            "number[0.0,2.0]",
            required=False,
            flag="--llm-exec-temp",
            default=0.0,
            description="Execution temperature.",
            envvar="KGOT_LLM_EXEC_TEMP",
        ),
    ]


def _single_params() -> list[dict[str, Any]]:
    return [
        _param(
            "problem",
            "string",
            required=True,
            flag="--problem/-p",
            description="Problem statement.",
        ),
        _param(
            "files",
            "array[string]",
            required=False,
            flag="--file/--files",
            default=[],
            description="Attachment paths in one directory.",
        ),
        *[
            {
                **param,
                "required": False,
                "default": None,
                "description": f"{param['description']} Overrides the global value.",
            }
            for param in _global_runtime_params()
        ],
        _param(
            "dry_run",
            "boolean",
            required=False,
            flag="--dry-run",
            default=False,
            description="Validate and preview without importing controllers.",
        ),
    ]


COMMANDS: list[dict[str, Any]] = [
    {
        "name": "single",
        "path": "kgot single",
        "description": "Solve one problem with the FalkorDB-backed KGoT controller.",
        "destructive": False,
        "exit_codes": {
            "0": "success",
            "2": "usage error",
            "3": "attachment not found",
            "4": "authentication error",
            "6": "rate limited",
            "7": "validation error",
            "8": "network error",
        },
        "params": _single_params(),
        "output": {
            "schema": "result.solution, result.iterations, result.graph_state, result.tool_calls"
        },
        "examples": [
            'kgot single --problem "What is a knowledge graph?" --dry-run',
            'kgot single --problem "What is a knowledge graph?"',
            'kgot single --problem "Summarize these files" --files ./data/a.txt --files ./data/b.txt',
        ],
    },
    {
        "name": "config",
        "path": "kgot config",
        "description": "Inspect effective non-secret runtime configuration.",
        "subcommands": [
            {
                "name": "show",
                "path": "kgot config show",
                "description": "Print sanitized configuration.",
                "params": [],
                "examples": ["kgot config show"],
            }
        ],
    },
    {
        "name": "tools",
        "path": "kgot tools",
        "description": "List supported offline tool configurations.",
        "subcommands": [
            {
                "name": "list",
                "path": "kgot tools list",
                "description": "List tool configuration modules.",
                "params": [],
                "examples": ["kgot tools list"],
            }
        ],
    },
    {
        "name": "skills",
        "path": "kgot skills",
        "description": "List or inspect agent skills shipped with the CLI.",
        "subcommands": [
            {
                "name": "list",
                "path": "kgot skills list",
                "description": "List skill names and commands.",
                "params": [],
                "examples": ["kgot skills list"],
            },
            {
                "name": "show",
                "path": "kgot skills show",
                "description": "Show one skill document.",
                "params": [
                    _param("name", "string", required=True, description="Skill name.")
                ],
                "examples": ["kgot skills show getting-started"],
            },
        ],
    },
    {
        "name": "issue",
        "path": "kgot issue",
        "description": "Store local structured feedback without an external service.",
        "subcommands": [
            {
                "name": "create",
                "path": "kgot issue create",
                "description": "Create an issue.",
                "params": [
                    _param(
                        "type",
                        "enum[bug,requirement,suggestion,bad-output]",
                        required=True,
                        flag="--type",
                        description="Issue category.",
                    ),
                    _param(
                        "message",
                        "string",
                        required=True,
                        flag="--message",
                        description="Issue description.",
                    ),
                    _param(
                        "context",
                        "string",
                        required=False,
                        flag="--context",
                        description="Optional JSON context object.",
                    ),
                    _param(
                        "exit_code",
                        "integer[0,255]",
                        required=False,
                        flag="--exit-code",
                        description="Exit code observed by the reporter.",
                    ),
                ],
                "examples": [
                    "kgot issue create --type bug --message 'single returned invalid JSON'"
                ],
            },
            {
                "name": "list",
                "path": "kgot issue list",
                "description": "List issues with pagination.",
                "params": [
                    _param(
                        "limit",
                        "integer[1,50]",
                        required=False,
                        flag="--limit",
                        default=20,
                        description="Maximum issues to return.",
                    ),
                    _param(
                        "offset",
                        "integer",
                        required=False,
                        flag="--offset",
                        default=0,
                        description="Number of issues to skip.",
                    ),
                ],
                "examples": ["kgot issue list --limit 20 --offset 0"],
            },
            {
                "name": "show",
                "path": "kgot issue show",
                "description": "Show one issue.",
                "params": [
                    _param(
                        "id", "string", required=True, description="Issue identifier."
                    )
                ],
                "examples": [
                    "kgot issue list",
                    "kgot issue show ISSUE-20260101T000000Z-ABCDEF12",
                ],
            },
            {
                "name": "close",
                "path": "kgot issue close",
                "description": "Close one issue.",
                "params": [
                    _param(
                        "id", "string", required=True, description="Issue identifier."
                    )
                ],
                "examples": ["kgot issue close ISSUE-20260101T000000Z-ABCDEF12"],
            },
        ],
    },
    {
        "name": "schema",
        "path": "kgot schema",
        "description": "Emit the complete command and parameter schema.",
        "params": [],
        "examples": ["kgot schema"],
    },
]


def root_help() -> dict[str, Any]:
    return {
        "help": fallback_brief(),
        "commands": COMMANDS,
        "output": {
            "default": "json",
            "human": "--human",
            "agent": "--agent",
            "quiet": "--quiet",
            "fields": "--fields name[,name]",
        },
        "exit_codes": {
            "0": "success",
            "1": "general error",
            "2": "usage error",
            "3": "resource not found",
            "4": "authentication error",
            "5": "conflict",
            "6": "rate limited",
            "7": "validation error",
            "8": "network error",
            "9": "schema error",
        },
    }


def command_help(name: str) -> dict[str, Any]:
    for command in COMMANDS:
        if command["name"] == name:
            return {"help": command["description"], "command": command}
    return {"help": f"No help metadata is registered for {name}.", "command": name}


def runtime_from_context(ctx: typer.Context) -> RuntimeConfig:
    values = (ctx.ensure_object(dict)).get("runtime")
    if isinstance(values, RuntimeConfig):
        return values
    if isinstance(values, dict):
        return RuntimeConfig(**values)
    return RuntimeConfig()


def output_from_context(
    ctx: typer.Context,
    *,
    human: bool = False,
    agent: bool = False,
    quiet: bool = False,
    fields: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    non_interactive: bool = False,
) -> dict[str, Any]:
    global_options = ctx.ensure_object(dict).get("output", {})
    resolved_human = bool(human or global_options.get("human", False))
    resolved_agent = bool(agent or global_options.get("agent", False))
    if resolved_human and resolved_agent:
        fail(
            "usage_error",
            "CONFLICTING_OUTPUT_MODES",
            "--human and --agent cannot be used together.",
            "Use --agent for JSON or --human for readable output, not both.",
            {},
            EXIT_USAGE,
        )
    _ = non_interactive or global_options.get("non_interactive", False)
    return {
        "human": resolved_human and not resolved_agent,
        "quiet": bool(quiet or global_options.get("quiet", False)),
        "fields": fields if fields is not None else global_options.get("fields"),
        "dry_run": bool(dry_run or global_options.get("dry_run", False)),
        "yes": bool(yes or global_options.get("yes", False)),
    }


def redact_uri(value: str) -> str:
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return "<invalid-uri>"


def sanitized_config(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "controller_choice": runtime.controller_choice,
        "tool_choice": runtime.tool_choice,
        "iterations": runtime.iterations,
        "snapshots": runtime.snapshots,
        "config_llm_path": runtime.config_llm_path,
        "logger_level": runtime.logger_level,
        "logger_file_mode": runtime.logger_file_mode,
        "statistics_file": runtime.statistics_file,
        "num_next_steps_decision": runtime.num_next_steps_decision,
        "max_retrieve_query_retry": runtime.max_retrieve_query_retry,
        "max_cypher_fixing_retry": runtime.max_cypher_fixing_retry,
        "max_final_solution_parsing": runtime.max_final_solution_parsing,
        "max_tool_retries": runtime.max_tool_retries,
        "max_llm_retries": runtime.max_llm_retries,
        "llm_plan": runtime.llm_plan,
        "llm_plan_temp": runtime.llm_plan_temp,
        "llm_exec": runtime.llm_exec,
        "llm_exec_temp": runtime.llm_exec_temp,
        "python_executor_uri": redact_uri(runtime.python_executor_uri),
        "falkordb_host": runtime.falkordb_host,
        "falkordb_port": runtime.falkordb_port,
        "falkordb_password_set": bool(runtime.falkordb_password),
        "mistral_api_key_set": bool(runtime.mistral_api_key),
    }


def validate_runtime(runtime: RuntimeConfig) -> RuntimeConfig:
    runtime = RuntimeConfig(
        **{
            **runtime.__dict__,
            "controller_choice": validate_choice(
                runtime.controller_choice,
                "controller_choice",
                {"directRetrieve", "queryRetrieve"},
            ),
            "tool_choice": validate_choice(
                runtime.tool_choice, "tool_choice", {"tools_v2_3"}
            ),
        }
    )
    validate_number(runtime.iterations, "iterations", 1, 20)
    validate_number(runtime.num_next_steps_decision, "num_next_steps_decision", 1, 100)
    for field in (
        "max_retrieve_query_retry",
        "max_cypher_fixing_retry",
        "max_final_solution_parsing",
        "max_tool_retries",
        "max_llm_retries",
    ):
        validate_number(getattr(runtime, field), field, 0, 100)
    validate_number(runtime.llm_plan_temp, "llm_plan_temp", 0.0, 2.0)
    validate_number(runtime.llm_exec_temp, "llm_exec_temp", 0.0, 2.0)
    if runtime.logger_file_mode not in {"a", "w", "x"}:
        fail(
            "validation_error",
            "INVALID_CHOICE",
            "logger_file_mode must be one of a, w, or x.",
            "Pass --logger-file-mode a, w, or x.",
            {"value": runtime.logger_file_mode},
            EXIT_VALIDATION,
        )
    for field in ("config_llm_path", "statistics_file"):
        validate_path(getattr(runtime, field), field)
    if runtime.snapshots:
        validate_path(runtime.snapshots, "snapshots")
    return runtime


def controller_kwargs(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "python_executor_uri": runtime.python_executor_uri,
        "llm_planning_model": runtime.llm_plan,
        "llm_planning_temperature": runtime.llm_plan_temp,
        "llm_execution_model": runtime.llm_exec,
        "llm_execution_temperature": runtime.llm_exec_temp,
        "max_iterations": runtime.iterations,
        "logger_level": runtime.logger_level,
        "logger_file_mode": runtime.logger_file_mode,
        "statistics_file_name": runtime.statistics_file,
        "db_choice": GRAPH_BACKEND,
        "controller_choice": runtime.controller_choice,
        "tool_choice": runtime.tool_choice,
        "config_llm_path": runtime.config_llm_path,
        "max_retrieve_query_retry": runtime.max_retrieve_query_retry,
        "max_cypher_fixing_retry": runtime.max_cypher_fixing_retry,
        "max_final_solution_parsing": runtime.max_final_solution_parsing,
        "max_tool_retries": runtime.max_tool_retries,
        "max_llm_retries": runtime.max_llm_retries,
        "num_next_steps_decision": runtime.num_next_steps_decision,
        "falkordb_host": runtime.falkordb_host,
        "falkordb_port": runtime.falkordb_port,
        "falkordb_password": runtime.falkordb_password,
        "mistral_api_key": runtime.mistral_api_key,
    }


def classify_exception(exc: Exception) -> CliError:
    message = str(exc).strip() or exc.__class__.__name__
    lower = message.lower()
    if any(
        token in lower
        for token in ("401", "unauthorized", "api key", "credential", "authentication")
    ):
        return CliError(
            "auth_error",
            "AUTH_ERROR",
            "The configured service rejected authentication.",
            "Set the required provider credential in the documented environment variable and retry.",
            {},
            EXIT_AUTH,
        )
    if any(
        token in lower
        for token in ("timeout", "connection", "network", "429", "rate limit")
    ):
        if "429" in lower or "rate limit" in lower:
            return CliError(
                "rate_limited",
                "RATE_LIMITED",
                "The upstream service rate-limited the request.",
                "Wait 60 seconds and retry.",
                {},
                EXIT_RATE_LIMITED,
            )
        return CliError(
            "network_error",
            "NETWORK_ERROR",
            "The upstream service could not be reached.",
            "Check FalkorDB and model service connectivity, then retry.",
            {},
            EXIT_NETWORK,
        )
    return CliError(
        "execution_error",
        "GENERAL_ERROR",
        "KGoT execution failed.",
        "Inspect FalkorDB connectivity, model credentials, and logs before retrying.",
        {"exception_type": exc.__class__.__name__},
        EXIT_GENERAL,
    )


def effective_runtime(base: RuntimeConfig, overrides: dict[str, Any]) -> RuntimeConfig:
    values = {key: value for key, value in overrides.items() if value is not None}
    return RuntimeConfig(**{**base.__dict__, **values})


def group_options(
    ctx: typer.Context,
    help_requested: bool,
    human: bool,
    agent: bool,
    quiet: bool,
    fields: str | None,
) -> dict[str, Any]:
    options = output_from_context(
        ctx, human=human, agent=agent, quiet=quiet, fields=fields
    )
    if help_requested or ctx.invoked_subcommand is None:
        emit_success(
            ctx.info_name or "kgot",
            {"help": command_help(ctx.info_name or "kgot"), "commands": COMMANDS},
            **options,
        )
    return options


__all__ = [
    "COMMANDS",
    "GRAPH_BACKEND",
    "RuntimeConfig",
    "agent_rules",
    "agent_skills",
    "classify_exception",
    "command_help",
    "controller_kwargs",
    "effective_runtime",
    "emit_success",
    "fail",
    "fallback_brief",
    "group_options",
    "output_from_context",
    "package_version",
    "read_agent_file",
    "redact_uri",
    "root_help",
    "runtime_from_context",
    "sanitized_config",
    "validate_runtime",
    "SCHEMA_VERSION",
]
