---
name: review
description: Project-aware code review - run pywatershed's mechanical convention checks on the diff, then the built-in code-review skill for correctness, and merge both into one findings-only report written to a .md file in the repo root. Use when the user invokes /review, optionally with a PR number, branch, or effort level (low/medium/high/max), e.g. "/review 407 high".
---

# Project-aware review

Two layers over one diff, merged into one report: (A) mechanical
checks of pywatershed's conventions — run as greps/scripts so they are
identical from review to review — and (B) the harness's built-in
`code-review` skill for correctness bugs and cleanups.

## Ground rules

- Findings only. Never apply fixes as part of a review; fixes are a
  separate, discussed step afterwards.
- The review assumes the test suite and the pre-commit hooks are run
  outside the review. Do not run tests or re-run anything pre-commit
  covers; the assumed items are listed as bullets up-front in step 1.
- The merged report is written to a markdown file (step 1 fixes the
  path; step 5 writes it). Invoking `/review` is permission for that
  one write; it is not covered by the no-state-changes rule below.
- You run read-only checks (greps, read-only git `diff`/`log`/`status`
  against the target) and invoke the built-in `code-review` skill.
  Invoking `/review` is permission for the agents that built-in
  spawns. Anything that creates, mutates, or destroys state still
  requires asking the human AND receiving explicit permission first.
- You cannot launch `/code-review ultra` yourself (cloud-run, billed,
  human-triggered only) — hand the human the command instead.

## Procedure

1. **Establish the target diff and the output file.** Default target:
   the current branch vs `develop` (vs `main` only for release/hotfix
   branches). A PR number or branch in the invocation overrides.

   The report file lives in the repo root, named `review_<PR#>.md` for
   a PR target or `review_<branch>.md` for a branch target. If that
   name is ambiguous (no PR number and an unwieldy branch name, say)
   or the file already exists, ask the human now — before running any
   checks — where the report should go and what to call it; do not
   wait until the end to sort this out.

   Before proceeding, state:
   - the target, the file count, and the report path;
   - the effort level, with a one-line reminder that the choice
     matters strongly — depth and token cost scale with it, and at
     `high` and above the built-in review fans out multiple internal
     reviewer agents (multi-agent token usage should be expected);
   - the assumptions, as a bulleted list of everything this review
     takes as already checked outside it:
     - the test suite (`autotest/ci_local.sh` / CI);
     - the pre-commit hooks, enumerated live from
       `.pre-commit-config.yaml` at review time so the list never
       drifts (currently: ruff check/format, blackdoc, doctoc,
       nbstripout, the security script).

2. **Codegraph context (optional).** Check whether the codegraph MCP
   server's tools are available (ToolSearch for `codegraph`). If they
   are not, tell the human the review is better with it and suggest
   installing/enabling it — https://github.com/colbymchenry/codegraph
   (registered in Claude as `codegraph serve --mcp`; new sessions see
   it, sessions older than the install do not) — then continue
   without it. If they are available, use read-only codegraph queries
   before reviewing to establish:
   - the blast radius of the diff: callers/dependents of every
     changed public symbol, including impact beyond the files the
     diff touches;
   - the type hierarchy around changed classes (bases and
     subclasses).
   Feed both into the correctness layer and summarize them briefly in
   the report.

3. **Mechanical convention checks (A).** Check only what the diff
   touches; report pass/fail per item with the offending lines quoted:
   - New public class/function: exported in `pywatershed/__init__.py`
     (and the subpackage `__init__.py`) and listed in a hand-written
     `doc/api/*.rst` autosummary.
   - User-facing change: has a `doc/whats-new.rst` entry with
     `(:pull:`XXX`)` or the real PR number.
   - `.github/workflows/ci.yaml` changed: `autotest/ci_local.sh`
     updated to match (job/test-file correspondence — a read of both
     files, never a run of either).
   - New domain-test CI job: carries the skeleton/full `if:` gate with
     a `ci-<token>` (an ungated job runs on every push; see
     DEVELOPER.md "CI").
   - New/changed conditional skip in a test: `--ignore`d in the broad
     CI steps or given a dedicated step whose `--control_pattern`
     avoids it (CI runs `--error-for-skips`).
   - Dependency change: `pyproject.toml` vs `environment.yml` vs
     `environment_w_jupyter.yml` kept consistent (conda names may
     differ, e.g. `epiweeks4cf`).
   - Docstrings (ruff's lint selection has no pydocstyle rules, so
     nothing else checks these): every new or signature-changed
     public class/function/method has a docstring whose documented
     parameters, returns, and raises match the final code; behavior
     changes update the docstring, not just the code. Style follows
     the surrounding module (Sphinx napoleon; Google `Args:` is the
     dominant style in the package).
   - Type hints: every new or signature-changed function/method is
     fully annotated — every parameter and the return type. This is
     an existence check, not an accuracy check (the repo has no
     mypy/static enforcement, so hint correctness is not chased);
     a missing annotation is a finding.

4. **Correctness review (B).** Invoke the built-in `code-review`
   skill on the same target at the requested effort level (default
   `high`). Note: at `high` and above the built-in skill fans out
   several parallel reviewer subagents and then verifies findings —
   one `/review` invocation, many agents inside. In the report,
   translate any agent-count language into plain confidence terms
   ("flagged independently by two of the parallel reviewers, so
   higher confidence"), so the human is not left wondering how many
   reviews they paid for.

5. **One merged report, written to the file from step 1.** Convention
   findings first (they are cheap to fix and objective), then
   correctness findings ranked by severity. For each finding quote the
   offending code, not a description of it. State explicitly when a
   layer found nothing. Write this report to the path fixed in step 1,
   show it in chat as well, and end by stating the report's path.
   For a large or high-stakes diff (release PRs, multi-hundred-line
   features), end with the optional escalation line for the human to
   run, e.g. `/code-review ultra 407` — a separate, deeper cloud
   review; do not attempt to run it yourself.
