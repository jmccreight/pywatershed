---
name: migrate-v2-v3
description: Migrate a user's project from pywatershed 2.x to 3.x following the portable guide in version_migration_guides/v2_to_v3.md - scan their code for the seven breaking changes, report file:line findings with fixes, and apply edits only with permission. Use when the user invokes /migrate-v2-v3, asks for help upgrading from pywatershed 2 to 3, or reports errors after upgrading.
---

# Migrating a project from pywatershed 2.x to 3.x

The complete instructions are the portable guide at
`version_migration_guides/v2_to_v3.md` (repo root). Read it fully and
follow its Procedure section. This stub adds only the
harness-specific points:

- Ask the user where their project lives before scanning; it is
  usually not this repository. Get permission before reading outside
  the working directory.
- Scan only git-tracked files in the user's project, per the guide.
- Report findings as `file:line` with the guide's fix; apply edits
  only after the user approves them.
- Do not re-derive the guide's delta from git tags or source diffs;
  the guide is authoritative for the 2.x -> 3.0.0 boundary. For
  changes after 3.0.0, consult `doc/whats-new.rst`.
