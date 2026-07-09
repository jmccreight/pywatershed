"""Tests for PRMSAtmosphereTranspFrostDynamic against GSFLOW output.

The dynamic frost transpiration period is validated against transp_on
output by GSFLOW for runs with dyn_fallfrost_flag and dyn_springfrost_flag
set, e.g.:

    pytest test_prms_atmosphere_transp_frost_dynamic.py \
        --domain fgr_ag_2yr --control_pattern analysis

A regression test also verifies that PRMSAtmosphereTranspFrostDynamic
reproduces PRMSAtmosphereTranspFrost exactly when no dynamic frost files
are supplied.

Solar tables are not part of the fgr_ag_2yr test data so they are computed
by PRMSSolarGeometry and passed in memory (they do not affect transp_on).

No tolerances are used in the comparisons: transp_on is a 0/1 integer flag
computed from integer solar-day comparisons with no floating-point
arithmetic, so exact equality is expected. GSFLOW's netCDF output stores
the flag as float64 (exact 0.0/1.0), hence the int32 casts before
comparison.
"""

import numpy as np
import pytest
import xarray as xr

from pywatershed.atmosphere.prms_atmosphere import (
    PRMSAtmosphereTranspFrost,
    PRMSAtmosphereTranspFrostDynamic,
)
from pywatershed.atmosphere.prms_solar_geometry import PRMSSolarGeometry
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.parameters import PrmsParameters
from pywatershed.utils.prms_dyn_param import (
    get_dynamic_param_files_from_control,
)


@pytest.fixture(scope="function")
def dyn_frost_files(simulation):
    """Dynamic frost file paths, skipping domains without dynamic frost."""
    active = get_dynamic_param_files_from_control(simulation["control_file"])
    if ("fallfrost_dynamic" not in active) or (
        "springfrost_dynamic" not in active
    ):
        pytest.skip(
            "Only testing PRMSAtmosphereTranspFrostDynamic for controls "
            "with dynamic fall and spring frost dates."
        )
    return {kk: vv["path"] for kk, vv in active.items()}


@pytest.fixture(scope="function")
def control(simulation, dyn_frost_files):
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    if "netcdf_output_dir" in ctl.options.keys():
        del ctl.options["netcdf_output_dir"]
    return ctl


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters(simulation, control):
    param_file = simulation["dir"] / control.options["parameter_file"]
    return PrmsParameters.load(param_file)


def _init_atm(
    Atm, simulation, control, discretization, parameters, **dyn_kwargs
):
    """Instantiate an atmosphere class with soltabs from PRMSSolarGeometry."""
    cbh_dir = simulation["dir"]
    solar = PRMSSolarGeometry(
        control, discretization=discretization, parameters=parameters
    )
    atm = Atm(
        control=control,
        discretization=discretization,
        parameters=parameters,
        prcp=cbh_dir / "prcp.nc",
        tmax=cbh_dir / "tmax.nc",
        tmin=cbh_dir / "tmin.nc",
        soltab_potsw=solar.soltab_potsw,
        soltab_horad_potsw=solar.soltab_horad_potsw,
        **dyn_kwargs,
    )
    return atm, solar


def test_compare_gsflow(
    simulation, control, discretization, parameters, dyn_frost_files
):
    """Compare dynamic transp_on with GSFLOW/PRMS transp_on output."""
    atm, solar = _init_atm(
        PRMSAtmosphereTranspFrostDynamic,
        simulation,
        control,
        discretization,
        parameters,
        fall_frost_dyn=dyn_frost_files["fallfrost_dynamic"],
        spring_frost_dyn=dyn_frost_files["springfrost_dynamic"],
    )

    # all variables are calculated for all times on the first advance
    control.advance()
    solar.advance()
    atm.advance()

    answer = xr.open_dataarray(simulation["output_dir"] / "transp_on.nc")
    answer_time = answer["time"].values.astype("datetime64[s]")
    assert answer_time.shape == atm._time.shape
    assert (answer_time == atm._time).all()

    np.testing.assert_array_equal(
        atm.transp_on.data.astype(np.int32),
        answer.values.astype(np.int32),
    )

    return


def test_static_regression(
    simulation, control, discretization, parameters, dyn_frost_files
):
    """Without dynamic frost files, dynamic == static class exactly."""
    atm_dyn, solar = _init_atm(
        PRMSAtmosphereTranspFrostDynamic,
        simulation,
        control,
        discretization,
        parameters,
    )
    atm_static, _ = _init_atm(
        PRMSAtmosphereTranspFrost,
        simulation,
        control,
        discretization,
        parameters,
    )

    control.advance()
    solar.advance()
    atm_dyn.advance()
    atm_static.advance()

    np.testing.assert_array_equal(
        atm_dyn.transp_on.data, atm_static.transp_on.data
    )

    return
