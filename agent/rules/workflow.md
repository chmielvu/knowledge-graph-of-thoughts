---
name: workflow
description: Discover, dry-run, then execute without interactive prompts.
---
1. Run `kgot --help` or `kgot schema` and parse the JSON response.
2. Run `kgot single --problem '...' --dry-run` with any attachment paths.
3. If the preview is correct, run the same command without `--dry-run`.
4. Branch on `error.kind` and the documented exit code; never parse prose logs. Use `--human` only for a human-facing invocation.