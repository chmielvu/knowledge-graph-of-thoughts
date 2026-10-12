"""Fail-fast input guardrails for the KGoT CLI."""

from __future__ import annotations

import re
from pathlib import Path

from cli.output import EXIT_NOT_FOUND, EXIT_USAGE, EXIT_VALIDATION, fail

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_SHELL_META = re.compile(r"[;|&$()]")
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
)
_SENSITIVE_NAMES = {".env", ".env.local", ".env.production", ".key", ".pem"}
_SENSITIVE_SUFFIXES = (".env", ".key", ".pem")
ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]{2,63}$")


def validate_text(
    value: str | None,
    field: str,
    *,
    allow_shell_meta: bool = True,
    max_length: int = 100_000,
) -> str:
    if value is None or not value.strip():
        fail(
            "validation_error",
            "MISSING_REQUIRED",
            f"{field} is required and must not be empty.",
            f"Pass a non-empty value with `--{field.replace('_', '-')}`.",
            {"field": field},
            EXIT_USAGE,
        )
    if len(value) > max_length:
        fail(
            "validation_error",
            "VALIDATION_ERROR",
            f"{field} exceeds the maximum length of {max_length} characters.",
            "Shorten the value and retry.",
            {"field": field, "max_length": max_length},
            EXIT_VALIDATION,
        )
    if _CONTROL_CHARS.search(value):
        fail(
            "validation_error",
            "VALIDATION_ERROR",
            f"{field} contains control characters.",
            "Remove control characters and retry.",
            {"field": field},
            EXIT_VALIDATION,
        )
    if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        fail(
            "validation_error",
            "SECRET_INPUT_REJECTED",
            f"{field} appears to contain a credential or private key.",
            "Remove the credential and use the documented environment variable instead.",
            {"field": field},
            EXIT_VALIDATION,
        )
    if not allow_shell_meta and _SHELL_META.search(value):
        fail(
            "validation_error",
            "UNSAFE_ARGUMENT",
            f"{field} contains shell metacharacters.",
            "Pass an opaque value without shell metacharacters.",
            {"field": field},
            EXIT_VALIDATION,
        )
    return value


def validate_opaque(value: str, field: str) -> str:
    return validate_text(value, field, allow_shell_meta=False, max_length=4_096)


def validate_choice(value: str, field: str, choices: set[str]) -> str:
    normalized = value if value in choices else value.lower()
    aliases = {choice.lower(): choice for choice in choices}
    normalized = aliases.get(normalized.lower(), normalized)
    if normalized not in choices:
        fail(
            "validation_error",
            "INVALID_CHOICE",
            f"Unsupported {field}: {value}.",
            f"Use one of: {', '.join(sorted(choices))}.",
            {"field": field, "value": value, "choices": sorted(choices)},
            EXIT_VALIDATION,
        )
    return normalized


def validate_number(
    value: int | float,
    field: str,
    minimum: int | float,
    maximum: int | float | None = None,
) -> None:
    if value < minimum or (maximum is not None and value > maximum):
        range_text = (
            f"at least {minimum}"
            if maximum is None
            else f"between {minimum} and {maximum}"
        )
        fail(
            "validation_error",
            "INVALID_RANGE",
            f"{field} must be {range_text}.",
            f"Pass a valid numeric value for `--{field.replace('_', '-')}`.",
            {"field": field, "value": value},
            EXIT_VALIDATION,
        )


def validate_path(raw: str, field: str, *, must_exist: bool = False) -> Path:
    value = validate_opaque(raw, field)
    path = Path(value)
    if ".." in path.parts or value.startswith(("../", "..\\")):
        fail(
            "validation_error",
            "PATH_TRAVERSAL_REJECTED",
            f"{field} contains path traversal.",
            "Use a path inside the working tree without `..` components.",
            {"field": field},
            EXIT_VALIDATION,
        )
    name = path.name.lower()
    if (
        name in _SENSITIVE_NAMES
        or name.startswith(".env")
        or name.endswith(_SENSITIVE_SUFFIXES)
    ):
        fail(
            "validation_error",
            "SENSITIVE_PATH_REJECTED",
            f"{field} points to a sensitive credential file.",
            "Do not pass .env, .key, or .pem files to the CLI.",
            {"field": field},
            EXIT_VALIDATION,
        )
    if must_exist and not path.is_file():
        fail(
            "not_found",
            "NOT_FOUND",
            f"File does not exist: {value}.",
            "Pass an existing readable attachment path.",
            {"field": field, "path": value},
            EXIT_NOT_FOUND,
        )
    return path
