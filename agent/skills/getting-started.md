---
name: getting-started
description: Discover the CLI and run a safe dry-run.
---
# Getting started

Use the following sequence:

```bash
kgot --help
kgot schema
kgot config show
kgot single --problem "your task" --dry-run
kgot single --problem "your task"
```

The default response is one JSON document on stdout. Errors are JSON on stderr and expose a stable `error.kind`, `error.code`, and exit code. Attachments must be existing non-sensitive files in one directory.

Use `--help` (and `<subcommand> --help`) to get exact usage and parameters.