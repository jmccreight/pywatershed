<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Run 1: GSFLOW (reference)](#run-1-gsflow-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Run 1: GSFLOW (reference)

GSFLOW 2.4.0 in PRMS-only mode — the reference run for the ag water-usage
comparison. See `../CLAUDE.md` for the full experiment description.

- **Binary:** `../../bin/gsflow_2.4.0_ifort_apple_silicon_dbl_prec` (this Mac).
- **Controls (version-controlled here):** two chained controls, adapted from
  the existing `fgr_ag_2yr` `spinup.control`/`analysis.control`:
  - `spinup_2000.control` — year 1 (2000-01-01 → 2000-12-31),
    `PRMSSoilzoneAg` config (`soilzone_aet_flag=0`, no iteration). Writes the
    restart `output_spinup/gsflow_ic_2000-12-31.ic`.
  - `analysis_2001.control` — year 2 (2001-01-01 → 2001-12-31),
    `PRMSSoilzoneAgObsET` config (`iter_aet_flag=1`, AET/PET cbh, dynamic
    ag-frac + frost). Reads that restart (`init_vars_from_file=1`).
- **Run from this dir.** Inputs are symlinked to `../../test_data/fgr_ag_2yr/`
  (`myparam.param`, `sf_data`, `*.cbh`, `dyn_ag_frac.param`, `*_frost.dyn`).
- **Outputs:** `output_spinup/` and `output_analysis/` (git-ignored).

Run both phases in order with `./run.sh` (spinup writes the IC, then analysis
reads it). Override the binary for another platform with
`GSFLOW=/path/to/exe ./run.sh`.

Verified: GSFLOW writes/reads `output_spinup/gsflow_ic_2000-12-31.ic`; the
analysis run reports `Restart File: 2000/01/01 - 2000/12/31`. The analysis run
emits many non-fatal `soil_lower exceeds soil_lower_stor_max` mass-balance
messages (small excesses) but still reaches normal completion — something to
check pywatershed reproduces.
