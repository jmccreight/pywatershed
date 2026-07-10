<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ag_comparison — running the experiments](#ag_comparison--running-the-experiments)
  - [Run 1 — GSFLOW reference](#run-1--gsflow-reference)
  - [Run 2 — full pywatershed](#run-2--full-pywatershed)
  - [Run 3 — pywatershed below snow, forced by GSFLOW outputs](#run-3--pywatershed-below-snow-forced-by-gsflow-outputs)
  - [Comparison and findings](#comparison-and-findings)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# ag_comparison — running the experiments

GSFLOW 2.4.0 (PRMS-only) vs. pywatershed "ag water usage" comparison on
the `fgr_ag_2yr` domain. Full background, design, status, and findings:
[CLAUDE.md](CLAUDE.md).

All commands below are run **from this directory** (`ag_comparison/`),
in an environment with pywatershed installed (all scripts resolve their
own paths, so they also work from anywhere). Each run is a year-1
spinup (2000, writes restart) chained to a year-2 analysis (2001, reads
restart). Later runs depend on earlier ones as noted.

## Run 1 — GSFLOW reference

```shell
./01_gsflow/run.sh
```

Runs both phases (spinup then analysis) with the GSFLOW 2.4.0 binary
from `../bin/` (override with `GSFLOW=/path/to/exe`). Outputs:
`01_gsflow/output_spinup/`, `01_gsflow/output_analysis/` (per-variable
CSVs).

## Run 2 — full pywatershed

```shell
python 02_pywatershed/run.py spinup
python 02_pywatershed/run.py analysis_dynamic_frost
python 02_pywatershed/run.py analysis_static_frost
```

Phases may also be run all at once (`python 02_pywatershed/run.py`).
The two analysis variants read the same spinup restart:
`analysis_dynamic_frost` is the GSFLOW-faithful configuration
(`PRMSAtmosphereTranspFrostDynamic` with the domain's `.dyn` files);
`analysis_static_frost` quantifies the effect of dynamic frost dates.
Outputs: `02_pywatershed/output_spinup/`,
`02_pywatershed/output_analysis_dynamic_frost/`,
`02_pywatershed/output_analysis_static_frost/`.

## Run 3 — pywatershed below snow, forced by GSFLOW outputs

Requires run 1 to have been run (its outputs are converted to run 3's
inputs). Re-run `convert` whenever run 1 is re-run.

```shell
python 03_pywatershed_below_snow/run.py convert
python 03_pywatershed_below_snow/run.py spinup
python 03_pywatershed_below_snow/run.py analysis
```

`convert` turns run 1's CSVs into NetCDF inputs
(`03_pywatershed_below_snow/gsflow_nc_{spinup,analysis}/`), including
derived variables. The model phases run only the below-snow process
chain (runoff, soilzone, groundwater, channel) — transpiration, snow,
and everything above arrive as data from GSFLOW. Outputs:
`03_pywatershed_below_snow/output_spinup/`,
`03_pywatershed_below_snow/output_analysis/`.

## Comparison and findings

```shell
python compare_runs.py
```

Computes Pearson r² and Nash-Sutcliffe efficiency (NSE) of each
pywatershed run against GSFLOW (run 1), over all day-hru pairs:
`hru_actet` for the spinup phase and `ag_irrigation_add` — the
estimated applied irrigation, the point of this configuration — for the
analysis phase. Results (2026-07-10):

| phase    | variable          | pywatershed run                | pearson r² | NSE      |
|----------|-------------------|--------------------------------|-----------:|---------:|
| spinup   | hru_actet         | run 2 (full pws)               |   0.996060 | 0.996043 |
| spinup   | hru_actet         | run 3 (below snow)             |   0.999996 | 0.999996 |
| analysis | ag_irrigation_add | run 2 (full pws, dyn frost)    |   0.993035 | 0.993035 |
| analysis | ag_irrigation_add | run 2 (full pws, static frost) |   0.236939 | −0.007992 |
| analysis | ag_irrigation_add | run 3 (below snow)             |   0.993389 | 0.993389 |

Findings:

1. **Dynamic frost dates are decisive for the irrigation estimate.**
   With static frost (pywatershed's only option before
   `PRMSAtmosphereTranspFrostDynamic`), `ag_irrigation_add` is no
   better than the GSFLOW mean (NSE ≈ 0); with dynamic frost it matches
   GSFLOW at r² = 0.993.
2. **With frost fixed, the full model is nearly as faithful as the
   GSFLOW-forced one** for `ag_irrigation_add` (0.9930 vs 0.9934) — the
   residual divergence in the target quantity is not dominated by
   snow/above-snow reproducibility; it lives in the below-snow chain or
   the AET iteration itself.
3. **For spinup AET the snow-dominance hypothesis holds**: forcing the
   below-snow chain with GSFLOW outputs raises r² from 0.996 to
   0.999996, so that residual was upstream of the soil zone.
