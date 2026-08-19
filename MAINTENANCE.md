<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Maintenance ledger](#maintenance-ledger)
  - [Open](#open)
    - [3.0.1 hotfix release (flopy pin unwind + pyPRMS floor)](#301-hotfix-release-flopy-pin-unwind--pyprms-floor)
    - [conda-forge feedstock: pyprms floor replaces packaging pin](#conda-forge-feedstock-pyprms-floor-replaces-packaging-pin)
    - [PR #407 (cascades port): gate the new sagehen CI jobs](#pr-407-cascades-port-gate-the-new-sagehen-ci-jobs)
    - [pyPRMS upstream: close PR #64, port its regression test](#pyprms-upstream-close-pr-64-port-its-regression-test)
    - [Publish gh-pages v3.0.0 extended release notes upstream](#publish-gh-pages-v300-extended-release-notes-upstream)
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

### conda-forge feedstock: pyprms floor replaces packaging pin

- **Blocked on:** nothing (pyprms 0.10.0 reached conda-forge 2026-08-18).
  Check: `https://api.anaconda.org/package/conda-forge/pyprms` versions,
  and PRs at `conda-forge/pywatershed-feedstock`.
- **Action:** the edit exists in a local clone
  (`recipe/meta.yaml`: `pyprms >=0.10.0` replacing `packaging <26.3`,
  build number 0 -> 1, branch `unpin_packaging`): push, open the PR,
  comment `@conda-forge-admin, please rerender`, merge on green. This is
  the last `packaging <26.3` pin anywhere.

### PR #407 (cascades port): gate the new sagehen CI jobs

- **Blocked on:** merging `develop` (with the skeleton/full CI gates,
  PR #408) into the `feat_cascades_port` branch. Check: PR #407 state and
  whether its `ci.yaml` sagehen jobs carry `if:` gates.
- **Action:** paste the domain-job `if:` gate (copy from any gated job in
  `ci.yaml`) into each new sagehen job with token `ci-sagehen`; then test
  with a push whose head commit message contains `ci-sagehen`. See
  DEVELOPER.md under "CI".

### pyPRMS upstream: close PR #64, port its regression test

- **Blocked on:** coordination with pyPRMS maintainer (pnorton-usgs).
  Check: state of DOI-USGS/pyPRMS PR #64.
- **Action:** close #64 (superseded by the higher-level fix in commit
  3650b285, released in 0.10.0) but port its regression test
  (`tests/func/test_Parameters.py::TestParametersSharedMetadata` — two
  `Parameters` instances from one shared `MetaData` dict).

### Publish gh-pages v3.0.0 extended release notes upstream

- **Blocked on:** James's push access to `DOI-USGS/pywatershed`
  `gh-pages` (org permission issue; the Actions "Run workflow" button is
  also missing upstream — possibly the same knot; ask org admins).
  Check: does `https://doi-usgs.github.io/pywatershed/` show the
  2026-07-13 v3.0.0 post?
- **Action:** `git push upstream gh-pages` (content is complete and
  previewed at `jmccreight.github.io/pywatershed`). Then verify the links
  in the GitHub 3.0.0 release body and `doc/index.rst` resolve. This is
  the last step (9) of the 3.0.0 release.

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

- 2026-08-18: PR #406 (pyPRMS>=0.10.0 floor replaces packaging<26.3 in
  pyproject + both env files) and PR #408 (CI skeleton/full split,
  ci-tokens, concurrency cancellation) merged to develop.
- 2026-08-18: pyPRMS 0.10.0 on PyPI and conda-forge; no pywatershed hot
  release needed (3.0.0's published metadata never pinned packaging, so
  fresh installs self-healed).
