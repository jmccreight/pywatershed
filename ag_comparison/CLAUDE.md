<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ag_comparison — GSFLOW 2.4.0 (PRMS-only) vs. pywatershed](#ag_comparison--gsflow-240-prms-only-vs-pywatershed)
  - [Status of this branch](#status-of-this-branch)
  - [The configuration under study: "ag water usage"](#the-configuration-under-study-ag-water-usage)
    - [Two phases per model run (chained via restart)](#two-phases-per-model-run-chained-via-restart)
  - [Domain: `fgr_ag_2yr`](#domain-fgr_ag_2yr)
    - [transp_frost on fgr_ag_2yr — RESOLVED](#transp_frost-on-fgr_ag_2yr--resolved)
  - [Existing tests covering this functionality](#existing-tests-covering-this-functionality)
  - [Original CONUS control files (`original_control_files/`)](#original-conus-control-files-original_control_files)
    - [Diffing controls — plan](#diffing-controls--plan)
  - [GSFLOW binary](#gsflow-binary)
  - [Experiment plan](#experiment-plan)
  - [Directory layout](#directory-layout)
    - [Working hypothesis](#working-hypothesis)
  - [Decisions (RESOLVED)](#decisions-resolved)
  - [Run status](#run-status)
  - [Open questions / TODO (to confirm with James)](#open-questions--todo-to-confirm-with-james)
  - [Key file map](#key-file-map)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# ag_comparison — GSFLOW 2.4.0 (PRMS-only) vs. pywatershed

Notes for Claude and for the team. This directory organizes experiments that
compare GSFLOW 2.4.0 (run in PRMS-only mode) "ag water usage" runs against the
equivalent pywatershed implementation.

> The parent repo's ground rules in `../CLAUDE.md` apply here too. In short:
> no running code (incl. git) without explicit permission; agree on a plan
> before many edits; short summaries by default; ask before spawning agents.

## Status of this branch

- Branch: `feat_ag_comparison`.
- **This branch is NOT intended to be merged** to `develop`/`main`. It exists to
  set up and run the comparison experiments and record findings.

## The configuration under study: "ag water usage"

The overall configuration estimates **agricultural water usage (applied
irrigation) per HRU** from *unmet actual ET*: the model compares its computed
agricultural AET against an "observed" AET per HRU and iteratively adds
irrigation to close the gap. This is a specific GSFLOW configuration; the
original CONUS control files are supplied in `original_control_files/` (see
below) and are adapted to the `fgr_ag_2yr` domain for these experiments.

### Two phases per model run (chained via restart)

Each of the three runs (below) is executed as **two chained sub-runs** — a
year-1 spinup that *writes* a restart, and a year-2 analysis that *reads* it.
This mirrors the original CONUS control pair.

1. **Spinup phase (year 1, 2000)** — no observed-ET iteration; writes restart.
   - pywatershed class: `PRMSSoilzoneAg`
     (`pywatershed/hydrology/prms_soilzone_ag.py`).
   - A simplified `PRMSSoilzoneAgObsET` with `iter_aet_flag=False`, no
     `aet_observed` input, no iterative AET matching.
   - GSFLOW analog: `soilzone_aet_flag=0`, `save_vars_to_file=1`,
     `init_vars_from_file=0` (see `original_control_files/nhm_ic.control`).
2. **Analysis phase (year 2, 2001)** — iterative AET matching (the actual
   water-usage estimate); reads restart.
   - pywatershed class: `PRMSSoilzoneAgObsET`
     (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py`).
   - Iterates `ag_irrigation_add` to match observed AET within
     `soilzone_aet_converge` (max `max_soilzone_ag_iter` iters, default 10).
     Only iterates when `iter_aet_flag=True` and `transp_on=True`.
   - GSFLOW analog: `iter_aet_flag=1`, `AET_cbh_file`/`PET_cbh_file` set,
     `init_vars_from_file=1` reading the year-1 IC, dynamic ag-frac and frost
     enabled (see `original_control_files/nhm_dynamic_2000_2020.control`).

Restart mechanics reference: `autotest/test_prms_soilzone_ag_restart.py`
(perfect-restart tests for both ag soilzone classes on `fgr_ag_2yr`) and
`autotest/test_nhm_restart.py` (per-process restart pattern; note its comment
that `transp_on` had to be promoted to a restart variable to pass daily
restart). Use `restart_write`/`restart_write_freq` on the year-1 process and
`restart_read` on year-2; `Control.edit_init_start_times` / `edit_end_time`
adjust the window.

Both pywatershed classes are based on GSFLOW 2.4.0 `soilzone_ag.f90` (see the
module docstring of `prms_soilzone_ag_obs_et.py` for the list of implemented vs.
neglected Fortran functionality — e.g. GLACIER, GSFLOW-active MODFLOW coupling,
cascades, and frozen-ground CFGI logic are neglected).

The configuration also uses the **`AtmosphereTranspFrost`** transpiration
module (`transp_module = transp_frost`).

## Domain: `fgr_ag_2yr`

- Obtained via `pywatershed.utils.addtl_domain_files`; already present in the
  repo at `test_data/fgr_ag_2yr` (a symlink to
  `pywatershed/data/pywatershed_addtl_domains/fgr_ag_2yr`, created by
  `autotest/ci_local.sh`).
- Simulation window in the current controls: **2000-01-01 to 2001-12-31** (2 yr).
- Key files already present in the domain dir:
  - Controls: `spinup.control`, `analysis.control`.
  - PRMS params: `myparam.param` (legacy); pywatershed per-process `.nc` params
    incl. `parameters_PRMSSoilzoneAg.nc`, `parameters_PRMSRunoffAg.nc`,
    `parameters_PRMSSnow.nc`, etc.
  - Observed/forcing: `aet_observed.nc`, `pet_observed.nc`,
    `actet_openet.cbh`, `potet_openet.cbh`, `prcp/tmax/tmin` (`.cbh` and `.nc`).
  - Dynamic ag frac: `ag_frac_static.nc`, `dyn_ag_frac.param`.
  - PRMS outputs already generated under `output_spinup/` and `output_analysis/`.
- Control facts (from `spinup.control`): `model_mode = PRMS5`,
  `soilzone_module = soilzone_ag`, `et_module = potet_jh`,
  `srunoff_module = srunoff_smidx`, `transp_module = transp_frost`,
  `strmflow_module = muskingum_mann`, `dprst_flag = 1`.

### transp_frost on fgr_ag_2yr — RESOLVED

Question raised: does `fgr_ag_2yr` have the `transp_frost` parameters?
**Yes.** Both `spinup.control` and `analysis.control` already set
`transp_module = transp_frost`, and `myparam.param` already contains the
required params (`tmax_index`, `fall_frost`, `spring_frost`, plus
`basin_fall_frost`/`basin_spring_frost`; `fall_frost.dyn`/`spring_frost.dyn`
also present). So no control/param edits are needed to enable transp_frost here.

Note: `transp_frost` is exercised by `test_prms_atmosphere_transp_frost.py` on
the `hru_1`, `drb_2yr`, and `ucb_2yr` domains in `ci_local.sh` (via
`--control_pattern=nhm_transp_frost.control`) — **not** on `fgr_ag_2yr` — but
the parameters/config do exist for `fgr_ag_2yr` as noted above.

## Existing tests covering this functionality

Run in `autotest/ci_local.sh` under the `fgr_ag_2yr` (`-f`) section, driven by
`--control_pattern=spinup.control` and `--control_pattern=analysis.control`:

- `test_prms_runoff_ag.py`, `test_prms_runoff_ag_restart.py`
- `test_prms_soilzone_ag.py`, `test_prms_soilzone_ag_restart.py`
- `test_prms_runoff_soilzone_ag.py`

`test_prms_runoff_soilzone_ag.py` tests **both** soilzone classes: its
`SoilzoneAg` fixture selects `PRMSSoilzoneAgObsET` when the control's
`iter_aet_flag` is set (analysis) and `PRMSSoilzoneAg` otherwise (spinup). It
only runs for domains whose `executable_desc` mentions GSFLOW.

Note (from `diff_controls.py`): the existing `fgr_ag_2yr` `spinup.control` and
`analysis.control` are **independent full-window runs**, not a chained
spinup→analysis — both have `init_vars_from_file=0`, so analysis does *not*
read the spinup IC. (Restart is exercised separately, in the `*_restart.py`
tests.) The CONUS originals *do* chain (`nhm_dynamic` reads `nhm_2000.ic`).
Our experiment therefore needs **new chained controls** (year-1 writes, year-2
reads), authored from the originals rather than reused from these test files.

Above-snow verification: `autotest/test_prms_above_snow.py` compares a
comprehensive suite of variables for the processes *above* snow
(SolarGeometry, Atmosphere, Canopy) against PRMS output — a useful template for
checking snow's inputs.

## Original CONUS control files (`original_control_files/`)

The two supplied originals *are* the write-restart → read-restart pair, over
CONUS 20/21-yr windows:

- `nhm_ic.control` — spinup / initial-conditions run, 1980–1999.
  `soilzone_aet_flag=0`, no `iter_aet_flag`, `save_vars_to_file=1` →
  writes `./input/nhm_2000.ic`. Params: `NHM.param`, `NHM_ag.param`,
  `frost_date_1980.param`.
- `nhm_dynamic_2000_2020.control` — analysis run, 2000–2020.
  `init_vars_from_file=1` reads `./input/nhm_2000.ic`; `iter_aet_flag=1`;
  `AET_module`/`PET_ag_module=climate_hru` with `AET_cbh_file=actet_openet.cbh`
  and `PET_cbh_file=potet_openet.cbh`; `dyn_ag_frac_flag=1`; dynamic spring/fall
  frost. Params: `NHM.param`, `NHM_ag.param`, `frost_date_2000.param`.

Both still carry the dated string `executable_desc = "GSFLOW version 2.3.0"`.

### Diffing controls — plan

Automated via `ag_comparison/diff_controls.py`. It is **self-contained**: the
comparisons live in the `COMPARISONS` list in the script; run it with no args
to (re)generate Markdown reports under `ag_comparison/diffs/`. Add a comparison
by appending a `(control_a, control_b, output_name)` tuple.

    python diff_controls.py

Core comparison uses `pyPRMS.ControlFile.diff` (builds each `ControlFile` via
`pp.ControlFile(filename=..., metadata=<extended>)`, following
`autotest/test_domain_subset.py`). pyPRMS's bundled metadata does **not** model
the GSFLOW ag / experiment control vars (e.g. `iter_aet_flag`, `AET_cbh_file`,
`forcing_check_flag`) and raises on any unknown var; it also treats `param_file`
as scalar, so it can't parse GSFLOW's multi-file form.

**Option A (chosen for now):** the script extends a deep copy of
`pws.constants.pyprms_meta` so every var reads. Since each control block
declares its value type and count, entries are synthesized from the files:

- unknown var → add `{datatype, context}` inferred from valuetype/numval;
- known scalar var that appears multi-valued (e.g. `param_file`) → promote its
  `context` to `array` (datetime `start/end_time` left alone).

Differing values are split into a **scalar** table and a **list** section that
shows added/removed elements (set differences) — so large output-var-name lists
(`nhruOutVar_names`, etc.) report only what changed, not full dumps.

Synthesized/promoted vars are listed in each report — documenting exactly what
pyPRMS doesn't yet model. The **first-party fix (Option B, later)** is to add
these vars to pyPRMS's `xml/control.xml` upstream. Control vars observed missing
from pyPRMS metadata across our four control files (candidate Option-B list):

- string scalars: `AET_cbh_file`, `PET_cbh_file`, `AET_module`,
  `PET_ag_module`, `ag_frac_dynamic`
- int scalars/flags: `iter_aet_flag`, `dyn_ag_frac_flag`, `forcing_check_flag`
- `param_file` is *known* to pyPRMS but modeled as scalar; GSFLOW's multi-file
  form needs it as an array (promoted at runtime here).

`ag_comparison/diffs/` is generated output — git-ignore it.

## GSFLOW binary

Reference GSFLOW is **2.4.0 (unreleased)**. Binaries live in `../bin/`:

- `gsflow_2.4.0_ifort_apple_silicon_dbl_prec` — use on this Mac (M-series).
- `gsflow_2.4.0_gfortran_linux_dbl_prec`, `..._windows_dbl_prec.exe` — other OSes.
- `..._apple_silicon_dbl_prec_og` — an "original" copy kept as backup.

The `executable_desc = "GSFLOW version 2.3.0"` string in the control files is
**stale**; the actual comparison uses the 2.4.0 binary above. (RESOLVED.)

## Experiment plan

Adapt the original CONUS "ag water usage" controls to `fgr_ag_2yr`, then run a
**1-year spinup (2000) followed by a 1-year analysis (2001)** — only 2 yr of
data available — three ways. Each run = two chained control files (year-1
writes restart, year-2 reads it).

1. **GSFLOW** (2.4.0, PRMS-only mode) — the reference.
2. **pywatershed** — full model.
3. **pywatershed below snow, driven by GSFLOW outputs** — replace PRMSSnow (and
   everything above it) with GSFLOW output as input to the below-snow processes.

The GSFLOW and pywatershed runs can likely share the same (PRMS-format) control
files, since pywatershed reads PRMS controls.

## Directory layout

Control files are copied in (small, version-controlled, diffable per run);
large forcing/parameter inputs are **symlinked** to `test_data/fgr_ag_2yr`
(itself a symlink into `pywatershed/data/pywatershed_addtl_domains`; the domain
data is downloaded via `addtl_domain_files`, not stored in the repo, and is at
negligible risk of on-disk edits). Each run's `output/` is git-ignored.

```
ag_comparison/
  CLAUDE.md
  diff_controls.py                 # control-file diff tool
  diffs/                           # generated reports (git-ignored)
  .gitignore                       # ignores diffs/ and */output/
  original_control_files/          # supplied CONUS originals (VC'd)
  01_gsflow/                       # run 1 — GSFLOW reference
  02_pywatershed/                  # run 2 — full pywatershed
  03_pywatershed_below_snow/       # run 3 — pywatershed below GSFLOW snow
```

Each run dir has a `README.md`, will hold two chained controls (year-1 writes
restart, year-2 reads), input symlinks, and a git-ignored `output/`.

### Working hypothesis

Divergence observed between GSFLOW and pywatershed is **dominated by
`PRMSSnow` reproducibility** issues. Test:

1. Confirm differences exist between run (1) GSFLOW and run (2) pywatershed.
2. Show that run (3) — pywatershed below snow driven by GSFLOW's snow and all
   above-snow inputs — eliminates the vast majority of those differences.
3. Also verify many of snow's *inputs* by comparing the suite of model outputs
   as done in `test_prms_above_snow.py`.

## Decisions (RESOLVED)

- **Simulation split.** Year 1 (2000) = spinup, writes restart; year 2 (2001) =
  analysis, reads restart. Two control files per run.
- **GSFLOW version.** Compare against `../bin/gsflow_2.4.0_ifort_apple_silicon_dbl_prec`
  (unreleased 2.4.0). The "2.3.0" `executable_desc` string is stale.
- **Original controls.** Supplied in `original_control_files/`
  (`nhm_ic.control`, `nhm_dynamic_2000_2020.control`).
- **Where outputs live.** Under `ag_comparison/`, one subdir per run; inputs
  version-controlled, `output/` git-ignored.

## Run status

- **Run 1 (GSFLOW): authored and executed OK.**
  `01_gsflow/spinup_2000.control` (year 1, writes
  `output_spinup/gsflow_ic_2000-12-31.ic`) and `01_gsflow/analysis_2001.control`
  (year 2, reads it). Both reach normal completion via `01_gsflow/run.sh`; the
  restart chain is confirmed (analysis reports `Restart File: 2000/01/01 -
  2000/12/31`). Diff of the two: `diffs/01_gsflow__spinup_vs_analysis.md`.
  - Note: the analysis phase emits many non-fatal `soil_lower exceeds
    soil_lower_stor_max` mass-balance messages (GSFLOW continues) — check
    whether pywatershed reproduces this.
- **Run 2 (pywatershed): driver written, not yet run.** `02_pywatershed/run.py`
  — two chained phases reusing run 1's controls, full-model restart via
  `control.options["restart_write"/"restart_read"]` (freq `"y"` → writes
  `2000-12-31-*.nc`; year-2 `init_time = start - 1 step = 2000-12-31` reads
  them). Based on `examples/10_ag_irrigation_use.ipynb` (OL/Analysis process
  lists, ag_frac handling) but with two corrections vs. the notebook:
  - Uses **`PRMSAtmosphereTranspFrost`** (not plain `PRMSAtmosphere`) to match
    GSFLOW's `transp_module=transp_frost`. It inherits restart variables
    `["tmax_sum", "transp_on"]` from `PRMSAtmosphere`, so transpiration state
    carries across the restart boundary. PRMS-legacy params come from
    `myparam.param` (has `fall_frost`/`spring_frost`/`tmax_index`), so the
    absent `parameters_PRMSAtmosphereTranspFrost.nc` isn't needed.
  - **Fidelity gap to resolve:** `PRMSAtmosphereTranspFrost.calc_transp_frost`
    uses **static** `fall_frost`/`spring_frost` (tiled over time). GSFLOW's
    year-2 analysis used **dynamic** frost (`dyn_*frost_flag=1`,
    `fall_frost.dyn`/`spring_frost.dyn`). Year-1 spinup is static in both, but
    2001 transp_on timing may diverge. pywatershed appears to have no dynamic
    frost support — TODO confirm and decide how to handle.
- **Run 3 (pywatershed below snow):** not started.

## Open questions / TODO (to confirm with James)

- **Run 1 first-run check.** Verify GSFLOW writes the restart to exactly
  `output_spinup/gsflow_ic_2000-12-31.ic`; if GSFLOW derives the name from
  dates, align `var_init_file` in `analysis_2001.control`.
- **Author runs 2 & 3 controls.** Chained year-1/year-2 controls for the
  pywatershed and below-snow runs (can likely reuse run 1's PRMS-format
  controls).
- **Restart var coverage.** Verify the ag soilzone (and other below-snow)
  processes restart correctly for our window (cf. the `transp_on` caveat in
  `test_nhm_restart.py`).
- **Input symlinks.** Create per-run symlinks to the needed
  `test_data/fgr_ag_2yr` inputs once the controls define what each run reads.
- **Option B (pyPRMS).** Upstream the missing control-var metadata (list in the
  diff section) into pyPRMS's `xml/control.xml`.

## Key file map

| Purpose | Path |
| --- | --- |
| Spinup class | `pywatershed/hydrology/prms_soilzone_ag.py` |
| Analysis class (+ iteration) | `pywatershed/hydrology/prms_soilzone_ag_obs_et.py` |
| Ag runoff | `pywatershed/hydrology/prms_runoff_ag.py` |
| Ag tests | `autotest/test_prms_runoff_soilzone_ag.py`, `test_prms_runoff_ag.py`, `test_prms_soilzone_ag.py` |
| Restart references | `autotest/test_prms_soilzone_ag_restart.py`, `autotest/test_nhm_restart.py` |
| Above-snow verification template | `autotest/test_prms_above_snow.py` |
| CI driver (fgr_ag_2yr `-f` section) | `autotest/ci_local.sh` (~L660+) |
| Domain data | `test_data/fgr_ag_2yr/` |
| GSFLOW 2.4.0 binaries | `bin/gsflow_2.4.0_*` |
| Original CONUS controls | `ag_comparison/original_control_files/` |
| Control diff script | `ag_comparison/diff_controls.py` |
