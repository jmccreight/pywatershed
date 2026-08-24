<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Maintenance ledger](#maintenance-ledger)
  - [Open](#open)
    - [3.0.1 hotfix release (flopy pin unwind + pyPRMS floor)](#301-hotfix-release-flopy-pin-unwind--pyprms-floor)
    - [pyPRMS upstream: close PR #64, port its regression test](#pyprms-upstream-close-pr-64-port-its-regression-test)
    - [Reconcile check_version.yaml with release_preflight.sh](#reconcile-check_versionyaml-with-release_preflightsh)
    - [Deliver CI-usage findings to org admins](#deliver-ci-usage-findings-to-org-admins)
  - [Done](#done)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Maintenance ledger

Maintenance todos that are blocked on external events or span
repositories, so they survive between sessions and maintainers. Each
item states what unblocks it as a condition that can be checked
mechanically (PyPI, conda-forge, or GitHub APIs). The `/maintenance`
Claude skill (`.claude/skills/maintenance/`) walks this file, checks
each condition live, and reports what is actionable; humans edit this
file like any other (PRs to `develop`).

Item format: a heading, then **Blocked on** (the condition and how to
check it), **Action** (what to do once unblocked), and optional
**Notes**. Move finished items to the Done section with a date.

## Open

### 3.0.1 hotfix release (flopy pin unwind + pyPRMS floor)

- **Blocked on:** a flopy release containing modflowpy/flopy PR #2730
  (wipe the dfn/toml dir in `generate_classes`). Check: latest version at
  `https://pypi.org/pypi/flopy/json` is newer than 3.10.0 (Feb 2026) and
  its changelog/commits include #2730.
- **Action:** patch release `3.0.1` from `main` per `.github/RELEASE.md`
  (the `/release` skill assists):
  - Replace `"flopy[codegen] @ git+https://github.com/modflowpy/flopy.git"`
    in `environment.yml` and `environment_w_jupyter.yml` with the released
    floor (e.g. `flopy[codegen]>=X`). Leave the `mpsplines` git pin alone —
    it is deliberate.
  - Add a `pyPRMS >=0.10.0` floor to `pyproject.toml` on the release
    branch: 3.0.0's published metadata has no floor, so installing an old
    pyPRMS (<=0.9.10) with packaging >=26.3 still breaks; develop already
    has the floor (PR #406).
  - Afterwards: flopy floor + build-number bump on the conda-forge
    pywatershed feedstock.
- **Notes:** conda-forge flopy 3.10.0 predates the fix, so pywatershed
  codegen paths (e.g. `MmrToMf6Dfw` class generation) are broken with
  released flopy until then; flopy imports lazily so this is not an
  import-time problem.

### pyPRMS upstream: close PR #64, port its regression test

- **Blocked on:** nothing — #64 was closed unmerged 2026-08-17; the
  regression test is not on pyPRMS main (verified 2026-08-19).
- **Action:** port #64's regression test to pyPRMS via a fresh PR
  (`tests/func/test_Parameters.py::TestParametersSharedMetadata` — two
  `Parameters` instances from one shared `MetaData` dict; the fix
  itself shipped in 0.10.0 via commit 3650b285).

### Reconcile check_version.yaml with release_preflight.sh

- **Blocked on:** nothing; low priority.
- **Action:** fold the legacy workflow's `doc/index.rst` major-release
  check (exit code 7) into `.github/scripts/release_preflight.sh`, or
  retire `check_version.yaml` in favor of the preflight + release.yaml
  checks.

### Deliver CI-usage findings to org admins

- **Blocked on:** nothing; conversational.
- **Action:** pywatershed is public and uses only standard runners, so
  its minutes are free and cannot consume the org quota (proof: when the
  org exhausted its minutes, private-repo Actions halted while
  pywatershed kept running). The billing usage CSV attributes metered
  minutes to the private repos that spent them. Org-shared runner
  concurrency was addressed by PR #408 (skeleton/full split +
  cancellation). Also raise the missing upstream push access /
  "Run workflow" button.

## Done

- 2026-08-19: conda-forge pywatershed 3.0.0 build 1 published (feedstock
  PR #14: `pyprms >=0.10.0` replaces `packaging <26.3`; verified on
  anaconda.org, uploaded 19:35 UTC). The last `packaging <26.3` pin
  anywhere is gone. Also: PR #407's sagehen ubuntu OOM resolved by
  serial pytest (`-n=1`) on Linux.
- 2026-08-19: gh-pages v3.0.0 extended release notes are live upstream
  (`doi-usgs.github.io/pywatershed`, 2026-07-13 post) and all 3.0.0
  release-body links resolve — step 9 done, the 3.0.0 release is
  complete. (Push-access / "Run workflow" org questions moved into the
  CI-usage item.)
- 2026-08-19: PR #407's new sagehen CI jobs carry the skeleton/full
  `if:` gates (token `ci-sagehen`); its remaining ubuntu CI failure is
  being handled on the branch.
- 2026-08-18: PR #406 (pyPRMS>=0.10.0 floor replaces packaging<26.3 in
  pyproject + both env files) and PR #408 (CI skeleton/full split,
  ci-tokens, concurrency cancellation) merged to develop.
- 2026-08-18: pyPRMS 0.10.0 on PyPI and conda-forge; no pywatershed hot
  release needed (3.0.0's published metadata never pinned packaging, so
  fresh installs self-healed).
