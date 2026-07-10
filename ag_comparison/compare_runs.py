#!/usr/bin/env python
"""Bottom-line comparison of the pywatershed runs against GSFLOW (run 1).

For each pywatershed run and phase, computes Pearson r^2 and
Nash-Sutcliffe efficiency (NSE) against run 1's GSFLOW output over all
day-hru pairs. The variables compared are the experiment's targets:

  - spinup: hru_actet (ag_irrigation_add is identically zero in spinup,
    which runs without the observed-AET iteration)
  - analysis: ag_irrigation_add (the estimated applied irrigation, the
    point of the "ag water usage" configuration)

Requires runs 1-3 to have been executed (see README.md). Run:

    python compare_runs.py
"""

import pathlib as pl

import numpy as np
import pandas as pd
import xarray as xr

HERE = pl.Path(__file__).resolve().parent

CASES = [
    # (phase, variable, label, pywatershed output dir)
    (
        "spinup",
        "hru_actet",
        "run 2 (full pws)",
        "02_pywatershed/output_spinup",
    ),
    (
        "spinup",
        "hru_actet",
        "run 3 (below snow)",
        "03_pywatershed_below_snow/output_spinup",
    ),
    (
        "analysis",
        "ag_irrigation_add",
        "run 2 (full pws, dyn frost)",
        "02_pywatershed/output_analysis_dynamic_frost",
    ),
    (
        "analysis",
        "ag_irrigation_add",
        "run 2 (full pws, static frost)",
        "02_pywatershed/output_analysis_static_frost",
    ),
    (
        "analysis",
        "ag_irrigation_add",
        "run 3 (below snow)",
        "03_pywatershed_below_snow/output_analysis",
    ),
]


def gsflow_csv(phase, var):
    """GSFLOW nhruOut CSV as a (time, nhru) array."""
    df = pd.read_csv(
        HERE / "01_gsflow" / f"output_{phase}" / f"{var}.csv",
        index_col=0,
        parse_dates=True,
    )
    return df.values


def pws_nc(run_dir, var):
    return xr.open_dataarray(HERE / run_dir / f"{var}.nc").values


def stats(pws_vals, gsflow_vals):
    pp, gg = pws_vals.ravel(), gsflow_vals.ravel()
    r = np.corrcoef(pp, gg)[0, 1]
    nse = 1 - np.sum((pp - gg) ** 2) / np.sum((gg - np.mean(gg)) ** 2)
    return r**2, nse


def main():
    header = (
        f"{'phase':9s} {'variable':18s} {'pywatershed run':32s} "
        f"{'pearson r^2':>12s} {'NSE':>10s}"
    )
    print(header)
    for phase, var, label, run_dir in CASES:
        r2, nse = stats(pws_nc(run_dir, var), gsflow_csv(phase, var))
        print(f"{phase:9s} {var:18s} {label:32s} {r2:12.6f} {nse:10.6f}")


if __name__ == "__main__":
    main()
