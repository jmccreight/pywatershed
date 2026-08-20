---
name: release
description: Assist a pywatershed release step by step, following .github/RELEASE.md. Use when the user asks to release pywatershed or invokes /release, optionally with the version, e.g. "/release 3.0.0".
---

# Assisting a pywatershed release

You assist the human releaser through the procedure in
`.github/RELEASE.md`, which is the source of truth — read it fully
before starting, along with `.github/workflows/release.yaml` (the
automation it describes).

## Ground rules for the assistant

- **The human runs every command that mutates state**: all git commands,
  `gh` commands that create/edit/publish anything, and pushes. You draft
  exact, paste-ready commands with the real version substituted, and you
  explain what each will do before handing it over.
- You run read-only verifications yourself (file greps, the preflight
  script, HTTP checks of PyPI and GitHub) and report results.
- Anything not explicitly yours above — in particular anything that
  creates, mutates, or destroys git state, or publishes — requires
  asking the human AND receiving explicit permission first; a drafted
  command is only ever executed by the human.
- Releases are irreversible at two points — merging to `main` and
  publishing the release (PyPI versions can never be re-uploaded).
  Before each, explicitly confirm the checks passed and the human is
  ready.
- Never suggest squash-merging the release PR or the post-release PR to
  `develop`. Merge commits only; this keeps `main` and `develop` from
  diverging.

## Pre-release content review (before RELEASE.md's steps)

Before starting the numbered RELEASE.md steps, review the release
*content* on the source branch. These are patterns that have actually
gone stale in past releases:

- **whats-new.rst placeholders**: grep the unreleased section for
  ``:pull:`XXX``` and `Author Name`/`username` stubs; fill real PR
  numbers (`curl -s "https://api.github.com/repos/DOI-USGS/pywatershed/pulls?state=closed&base=develop&per_page=50"`
  maps branches to numbers). Check every entry has a `(:pull:...)`.
- **whats-new.rst completeness**: compare merged PRs since the last
  release against the entries; flag merged PRs with no entry (CI-only
  changes may be skipped deliberately — ask).
- **whats-new.rst anchor**: the new section needs a
  `.. _whats-new.X.Y.Z:` label above its heading, matching prior
  sections.
- **API doc coverage**: every new public class/function exported from
  `pywatershed/__init__.py` should appear in a hand-written
  `doc/api/*.rst` autosummary (grep, excluding `doc/api/generated/`,
  which is gitignored build output and often stale locally).
- **Packaging consistency**: reconcile `pyproject.toml` dependencies
  vs `environment.yml`/`environment_w_jupyter.yml` (conda names may
  differ, e.g. `epiweeks4cf`); check python version pins vs
  `requires-python` and classifiers.
- **No development pins at release**: `git+...` URLs or unreleased
  branches in `environment.yml`/`pyproject.toml` (e.g. a temporary
  `flopy @ git+...develop` workaround) will be captured in the frozen
  release envs — confirm each is intentional or replace with a
  released version.
- **CITATION.cff**: `cff-version:` is the CFF *schema* version and
  must stay `1.2.0` — do not bump it with the package; `version:` and
  `date-released:` are updated in RELEASE.md step 4 (preflight checks
  them).
- **code.json**: updated in RELEASE.md step 4 (`version`,
  `downloadURL`, `metadataLastUpdated`, `status`), but verify — no
  automated check covers this file and it goes stale silently.
- **conda recipe drift**: diff `pyproject.toml` dependencies and
  `requires-python` against the last release NOW — after publication
  they must be mirrored into the conda-forge feedstock (RELEASE.md
  step 10), and any dependency not yet on conda-forge needs a
  staged-recipes submission, which has days of review lead time
  (pyPRMS blocked the 3.0.0 conda package this way).

## Protocol

1. **Establish the release**: the version (from the invocation or ask),
   whether it is major/minor/patch, and therefore the source branch
   (`develop` for major/minor, `main` for patch). State the plan in one
   short paragraph mirroring the example at the top of RELEASE.md's
   step-by-step section.

2. **Walk RELEASE.md's numbered steps in order.** For each step:
   - Say which step you are on and what it accomplishes.
   - Draft the commands or file edits it requires. You may make the
     file edits of step 4 (whats-new heading and date, CITATION.cff,
     README disclaimer) directly when asked; the human commits them.
   - For a major release, ask for the new provisional DOI explicitly
     and early (step 1, not step 4) — only the human can obtain it
     from USGS, retrieval may need the VPN, and no automated check
     will catch a stale DOI. It appears in three places: the
     `README.md` badge, the `README.md` "How to Cite" line, and
     `CITATION.cff` `identifiers:`.
   - Verify before moving on. In particular:
     - After step 4: run `.github/scripts/release_preflight.sh` and
       show the result; do not proceed until it passes.
     - At step 5: confirm both CI and the release checks are green on
       the PR, and offer to review the `dist` artifact contents.
     - At step 6: restate merge-not-squash before the human merges.
     - At step 7 (after publishing): verify from outside —
       `curl -s https://pypi.org/pypi/pywatershed/json` shows the new
       version; the GitHub release exists, is not a draft, and has the
       three `environment_frozen_*.yml` assets
       (`curl -s https://api.github.com/repos/DOI-USGS/pywatershed/releases/latest`).
     - At step 8: check the post-release branch sets the `.dev0`
       version and the new "(Unreleased)" whats-new section before the
       PR to `develop` is opened.

3. **If anything fails**, consult RELEASE.md's "If something goes
   wrong" section and present the options there rather than improvising;
   the safe recovery paths are already worked out.

4. **At the end**, summarize what was released, the tag, the PyPI URL,
   and confirm `develop` carries the next `.dev0` version.
