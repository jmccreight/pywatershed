<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Maintenance ledger](#maintenance-ledger)
  - [Open](#open)
    - [3.0.1 hotfix release (flopy pin unwind + pyPRMS floor)](#301-hotfix-release-flopy-pin-unwind--pyprms-floor)
    - [pyPRMS upstream: close PR #64, port its regression test](#pyprms-upstream-close-pr-64-port-its-regression-test)
    - [Reconcile check_version.yaml with release_preflight.sh](#reconcile-check_versionyaml-with-release_preflightsh)
    - [Retire the GSFLOW ifort binary](#retire-the-gsflow-ifort-binary)
    - [Deliver CI-usage findings to org admins](#deliver-ci-usage-findings-to-org-admins)
    - [PRMSCanopy grass interception gate: pkwater_ante, not pk_ice + freeh2o](#prmscanopy-grass-interception-gate-pkwater_ante-not-pk_ice--freeh2o)
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

### Retire the GSFLOW ifort binary

- **Blocked on:** nothing external -- this is the follow-up half of the
  ifort retirement. Check: `git ls-files bin/` still lists
  `gsflow_2.4.0_ifort_apple_silicon_dbl_prec`.
- **Action:** decide what replaces it. GSFLOW source is not in this
  repository (`gsflow_src/` is gitignored), so unlike PRMS it cannot be
  compiled on demand; it is the last intel-built artifact here, and it is
  an **x86_64** binary, so on Apple Silicon it runs under Rosetta 2 --
  including on the `macos-latest` runners for the three
  `test_fgr_ag_2yr_*` jobs, which are the only consumers. Options:
  vendor/point at a GSFLOW source and build it with gfortran the way PRMS
  now is; ship a gfortran-built arm64 GSFLOW binary; or drop the binary
  and remove `macos-latest` from the `test_fgr_ag_2yr_*` matrices.
- **Notes:** `bin/gsflow_2.4.0_ifort_apple_silicon_dbl_prec_og` is
  byte-identical to the non-`_og` file (verified with `cmp`) and is
  referenced nowhere -- 11.9 MB of pure duplicate to delete whichever way
  the decision goes. The PRMS half of this work (binaries compiled on
  demand from `prms_src/`, ifort and Intel MacOS dropped, `m1` -> `mac_arm`)
  landed on branch `feat_drop_ifort`. That compile-on-demand scheme is an
  accepted intermediate step: the intended end state is to move the
  PRMS/GSFLOW sources into a separate "oracle" repository that publishes
  binaries for pywatershed to download, which would settle this item too.
  Prefer a cheap stopgap here over a design that assumes the sources stay
  in this repository.

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

### PRMSCanopy grass interception gate: pkwater_ante, not pk_ice + freeh2o

- **Blocked on:** nothing; this is a pywatershed port-fidelity bug, found
  2026-08-25 while retiring ifort (branch `feat_drop_ifort`).
- **Action:** in `PRMSCanopy`, take `pkwater_ante` as an input in place of
  `pk_ice_prev` and `freeh2o_prev`, and gate grass rain interception on
  `pkwater_ante[i] < dnearzero` (`prms_canopy.py:521`). `pkwater_ante` is
  already a PRMS output (declared `snowcomp.f90:346-349`, assigned
  `Pkwater_ante = Pkwater_equiv` at `snowcomp.f90:946`) and already has
  metadata (`variables.yaml:2663`), so no new diagnostic variable is
  needed. `PRMSSnow` must also declare `pkwater_ante` so coupled models
  can supply it; PRMS declares it in snowcomp, so that is faithful to the
  original design. Regenerate test data afterward: answers change wherever
  the gate flips.
- **Notes:** `intcp.f90:416` gates grass interception on
  `Pkwater_equiv(i) < DNEARZERO` (2.23e-16); `intcp` runs before
  `snowcomp`, so the value it sees is the previous step's, which is what
  the `pkwater_ante` output holds. pywatershed instead reconstructs the
  pack as `pk_ice_prev + freeh2o_prev`. Those are not the same number:
  PRMS carries `Pkwater_equiv` as its own accumulated double rather than
  recomputing it as the sum, and in the vanishing tail of a melting pack
  they diverge by orders of magnitude. At `ucb_2yr:nhm` HRU 257, step 652
  (1980-10-14), `pkwater_ante = 1.63e-14` while
  `pk_ice_prev + freeh2o_prev = 1.59e-16` -- straddling the 2.23e-16
  threshold. PRMS passes the day's rain through; pywatershed intercepts
  `srain_intcp = 0.013033`, so `net_rain` differs by
  `covden_sum * srain_intcp = 0.001272829` on that day and the two agree
  again the next. That fails `test_prms_canopy.py`,
  `test_prms_above_snow.py`, `test_prms_et_canopy.py`, and
  `test_prms_et_can_runoff.py` on macOS while Linux passes, because the
  residuals left by each build land on opposite sides of the threshold.
  The bug is latent and long-standing; the gfortran mac binary just rolled
  the dice differently than the ifort one did. The comment at
  `prms_canopy.py:519` already quotes the PRMS gate as
  `pkwater_ante(i)<dnearzero`, so the substitution was known when it was
  made. Caveat: this restores fidelity for the single-process comparisons,
  which read PRMS's own `pkwater_ante`. Fully coupled pywatershed runs
  still depend on `PRMSSnow` reproducing PRMS's residual tail to ~1e-16,
  which it almost certainly does not -- `snowcomp` is full of similar
  thresholds and `test_prms_snow.py` only checks 1e-3. Making `PRMSSnow`
  abide by the same standard as the other processes is a much larger lift
  and a separate item.

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
