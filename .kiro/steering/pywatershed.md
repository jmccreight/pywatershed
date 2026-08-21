<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [pywatershed project guidance](#pywatershed-project-guidance)
  - [Mechanism translations](#mechanism-translations)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# pywatershed project guidance

`CLAUDE.md` at the repo root and the skills in `.claude/skills/*/SKILL.md` are
addressed to you. Read `CLAUDE.md` before doing project work and behave exactly
as a Claude Code session would: every instruction in those files applies here in
full, including ones that say "Claude". There are no Kiro exemptions. `CLAUDE.md`
is the canonical guidance — testing (`autotest/`, domains, CI gates), conventions
(ruff, `doc/whats-new.rst`, public exports, PR bodies), branch and release
policy. Edit that file, not this one.

Only the mechanism names differ, and that never changes what you must do. A
mechanism mismatch is not permission to skip, soften, or defer an instruction.
Translate it and carry it out; if some instruction has no Kiro equivalent at all,
do the closest thing and say so in your reply rather than silently dropping it.

## Mechanism translations

- Skills are Claude Code slash commands; in Kiro they are manual-inclusion
  steering with the same names: `/release` → `#release`, `/ci-triage` →
  `#ci-triage`, `/maintenance` → `#maintenance`. When naming them to the user,
  use the `#` form.
- The `@~/.claude/pywatershed-instructions.md` import at the end of `CLAUDE.md`
  does nothing in Kiro. Personal untracked instructions belong in
  `~/.kiro/steering/`, which loads for every project.
