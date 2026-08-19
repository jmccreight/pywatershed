<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [pywatershed guidance for Claude](#pywatershed-guidance-for-claude)
  - [Testing](#testing)
  - [Conventions](#conventions)
  - [Branches and releases](#branches-and-releases)
  - [Skills](#skills)
  - [Personal instructions](#personal-instructions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# pywatershed guidance for Claude

pywatershed is a Python package for hydrologic modeling, a
reimplementation of PRMS process representations (see README.md).

## Testing

- Tests live in `autotest/` and are run from that directory. Tests using
  the `simulation` fixture require `--domain` (e.g.
  `pytest test_prms_atmosphere.py --domain drb_2yr`) and optionally
  `--control_pattern`; there is no default domain.
- Domain test data must be generated before testing; see DEVELOPER.md.
- `autotest/ci_local.sh` runs the CI suites locally and must be kept in
  sync with `.github/workflows/ci.yaml` (which test files each job runs
  or ignores).
- CI uses `--error-for-skips`: a test that skips conditionally must be
  either `--ignore`d in the broad CI steps or run in a dedicated step
  whose `--control_pattern` avoids the skip.
- Every domain test job in `ci.yaml` carries an `if:` gate (the
  skeleton/full split). A new domain job must copy the gate from an
  existing domain job, with an appropriate `ci-<token>` — an ungated job
  runs on every push to every branch. Watch for this especially when
  merging develop into branches that predate the gates. See DEVELOPER.md
  under "CI" for the strategy and token semantics.

## Conventions

- Lint and format with `ruff check .` and `ruff format .` (pre-commit
  also runs doctoc for markdown TOCs).
- User-facing changes get an entry in `doc/whats-new.rst`; use
  `(:pull:`XXX`)` as the placeholder until the PR number exists.
- New public classes are exported in `pywatershed/__init__.py` (and the
  subpackage `__init__.py`) and listed in `doc/api/*.rst`.
- When drafting a PR body, read `.github/PULL_REQUEST_TEMPLATE.md`
  (fresh each time — it evolves) and conform to it: summary on top,
  then its checklist with irrelevant items removed and applicable ones
  checked, keeping its Docs section.

## Branches and releases

- `main` always holds the latest release; `develop` holds the next one
  and is the base for feature branches.
- Flows between `main` and `develop` use merge commits, never squash.
- Releases follow `.github/RELEASE.md`; the `/release` skill assists
  step by step. `.github/scripts/release_preflight.sh` checks release
  version consistency locally and in CI.

## Skills

Project-specific skills live in `.claude/skills/` and are invoked as
slash commands in a Claude session. Keep this list current when adding
skills:

- `/release` — assists a pywatershed release step by step, following
  `.github/RELEASE.md`: drafts the commands for the human to run,
  runs the checks, and verifies each stage (PyPI, tag, release assets).
- `/ci-triage` — triages failing GitHub Actions runs on a PR or branch
  via the public API: stale-run vs real-failure verdict, annotations
  when logs are locked, known failure signatures, local repro steps.
- `/maintenance` — walks the `MAINTENANCE.md` ledger (repo root) of
  externally-blocked and cross-repo todos, checks each blocked-on
  condition live (PyPI, conda-forge, GitHub APIs), reports what is
  actionable, and updates the ledger.
- `/review` — project-aware code review of a diff (current branch vs
  develop, or a PR number): pywatershed's mechanical convention checks
  plus the harness's built-in `code-review` for correctness, merged
  into one findings-only report; offers `/code-review ultra` for big
  diffs.

Claude: in your first reply of a session, briefly show the user this
bullet list of available skills (many users don't know skills exist).
Keep it short and then address their request; don't let the list crowd
out their actual question.

## Personal instructions

Individual contributors may keep personal instructions (untracked) at
the path below; if the file does not exist the import is skipped.

@~/.claude/pywatershed-instructions.md
