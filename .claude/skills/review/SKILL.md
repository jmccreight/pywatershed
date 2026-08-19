---
name: review
description: Project-aware code review - run pywatershed's mechanical convention checks on the diff, then the built-in code-review skill for correctness, and merge both into one findings-only report. Use when the user invokes /review, optionally with a PR number, branch, or effort level (low/medium/high/max), e.g. "/review 407 high".
---

# Project-aware review

Two layers over one diff, merged into one report: (A) mechanical
checks of pywatershed's conventions — run as greps/scripts so they are
identical from review to review — and (B) the harness's built-in
`code-review` skill for correctness bugs and cleanups.

## Ground rules

- Findings only. Never apply fixes as part of a review; fixes are a
  separate, discussed step afterwards.
- You run read-only checks (greps, read-only git `diff`/`log`/`status`
  against the target, `ruff check`/`ruff format --check`) and invoke
  the built-in `code-review` skill. Invoking `/review` is permission
  for the agents that built-in spawns. Anything that creates, mutates,
  or destroys state still requires asking the human AND receiving
  explicit permission first.
- You cannot launch `/code-review ultra` yourself (cloud-run, billed,
  human-triggered only) — hand the human the command instead.

## Procedure

1. **Establish the target diff.** Default: the current branch vs
   `develop` (vs `main` only for release/hotfix branches). A PR number
   or branch in the invocation overrides. State the target and the
   file count before proceeding.

2. **Mechanical convention checks (A).** Check only what the diff
   touches; report pass/fail per item with the offending lines quoted:
   - New public class/function: exported in `pywatershed/__init__.py`
     (and the subpackage `__init__.py`) and listed in a hand-written
     `doc/api/*.rst` autosummary.
   - User-facing change: has a `doc/whats-new.rst` entry with
     `(:pull:`XXX`)` or the real PR number.
   - `.github/workflows/ci.yaml` changed: `autotest/ci_local.sh`
     updated to match (job/test-file correspondence).
   - New domain-test CI job: carries the skeleton/full `if:` gate with
     a `ci-<token>` (an ungated job runs on every push; see
     DEVELOPER.md "CI").
   - New/changed conditional skip in a test: `--ignore`d in the broad
     CI steps or given a dedicated step whose `--control_pattern`
     avoids it (CI runs `--error-for-skips`).
   - Dependency change: `pyproject.toml` vs `environment.yml` vs
     `environment_w_jupyter.yml` kept consistent (conda names may
     differ, e.g. `epiweeks4cf`).
   - `ruff check` and `ruff format --check` clean on the changed files.

3. **Correctness review (B).** Invoke the built-in `code-review`
   skill on the same target at the requested effort level (default
   `high`).

4. **One merged report.** Convention findings first (they are cheap
   to fix and objective), then correctness findings ranked by
   severity. For each finding quote the offending code, not a
   description of it. State explicitly when a layer found nothing.
   For a large or high-stakes diff (release PRs, multi-hundred-line
   features), end with the optional escalation line for the human to
   run, e.g. `/code-review ultra 407` — a separate, deeper cloud
   review; do not attempt to run it yourself.
