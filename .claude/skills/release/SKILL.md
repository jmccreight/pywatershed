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
- Releases are irreversible at two points — merging to `main` and
  publishing the release (PyPI versions can never be re-uploaded).
  Before each, explicitly confirm the checks passed and the human is
  ready.
- Never suggest squash-merging the release PR or the post-release PR to
  `develop`. Merge commits only; this keeps `main` and `develop` from
  diverging.

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
