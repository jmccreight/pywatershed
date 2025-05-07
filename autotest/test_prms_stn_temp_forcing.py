import pytest
from utils_compare import compare_in_memory

from pywatershed.atmosphere.prms_station_temp_forcing import (
    PRMSStationTempForcing,
)
from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.parameters import PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = False
do_compare_in_memory = True

rtol = atol = 1.0e-10


@pytest.fixture(scope="function")
def control(simulation):
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    del ctl.options["netcdf_output_dir"]
    return ctl


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters(simulation, control, request):
    # if request.param == "params_one":
    param_file = simulation["dir"] / control.options["parameter_file"]
    params = PrmsParameters.load(param_file)
    # else:
    #     param_file = simulation["dir"] / "parameters_PRMSSolarGeometry.nc"
    #     params = PrmsParameters.from_netcdf(param_file)
    return params


def test_compare_prms(
    simulation, control, discretization, parameters, tmp_path
):
    data_file = simulation["dir"] / control.options["data_file"]
    stns_dict = {}
    stn_vars = ["tmax", "tmin"]
    for vv in stn_vars:
        stns_dict[vv] = adapter_factory(
            var=data_file,
            variable_name=vv,
            control=control,
        )

    # DEAL/CHECK missing values, the file has both -901.0 and -9999.0

    temp_forcings = PRMSStationTempForcing(
        control,
        discretization=discretization,
        parameters=parameters,
        tmax_sta=stns_dict["tmax"],
        tmin_sta=stns_dict["tmin"],
    )

    check_vars = ["tmax", "tmin"]

    if do_compare_in_memory:
        answers = {}
        for var in check_vars:
            var_pth = simulation["dir"] / f"{var}.nc"
            answers[var] = adapter_factory(
                var_pth, variable_name=var, control=control
            )

    for ii in range(control.n_times):
        control.advance()
        temp_forcings.advance()
        temp_forcings.calculate(1.0)

        if do_compare_in_memory:
            for var in answers.values():
                var.advance()
            # <
            compare_in_memory(
                temp_forcings,
                answers,
                atol=atol,
                rtol=rtol,
                skip_missing_ans=True,
                fail_after_all_vars=False,
            )

    # test writing netcdf outputs.
    # if do_compare_output_files:
    #     compare_netcdfs(
    #         PRMSSolarGeometry.get_variables(),
    #         tmp_path,
    #         output_dir,
    #         atol=atol,
    #         rtol=rtol,
    #     )

    return
