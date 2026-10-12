---
name: writeback
description: Use the local issue command to report bugs and bad output.
---
Report a reproducible problem with `kgot issue create --type bug --message '...' --context '{...}'`. Use `bad-output` when the JSON contract is wrong, `requirement` for missing capability, and `suggestion` for non-blocking improvements. Issue data is stored locally in `.kgot/issues.json`; do not include credentials or private keys.