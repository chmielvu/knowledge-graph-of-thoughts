# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Main authors: Lorenzo Paleari, Ales Kubicek, Andrea Jiang

"""Compatibility entry point for the agent-friendly Typer CLI."""

from __future__ import annotations

from typing import Sequence

from kgot.cli import app, main as _run_cli

__all__ = ["app", "main"]


def main(args: Sequence[str] | None = None) -> int:
    """Run the Typer CLI while preserving the historical import path."""
    return _run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
