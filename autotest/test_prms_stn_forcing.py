import pytest
from utils_compare import compare_in_memory

from pywatershed.atmosphere.prms_station_forcings import (
    PRMSStationPrecipForcing,
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
    if (
        "temp_module" not in ctl.options.keys()
        or "precip_module" not in ctl.options.keys()
    ) or (
        ctl.options["temp_module"] != "temp_1sta"
        or ctl.options["precip_module"] != "precip_1sta"
    ):
        pytest.skip("The configuraiton does not use temp AND precip 1sta: ")
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


def test_compare_prms_temp_1sta(
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


def test_compare_prms_prcp_1sta(
    simulation, control, discretization, parameters, tmp_path
):
    data_file = simulation["dir"] / control.options["data_file"]

    # temp first
    stns_dict = {}
    stn_vars = ["tmax", "tmin", "precip"]
    for vv in stn_vars:
        stns_dict[vv] = adapter_factory(
            var=data_file,
            variable_name=vv,
            control=control,
        )

    temp_forcings = PRMSStationTempForcing(
        control,
        discretization=discretization,
        parameters=parameters,
        tmax_sta=stns_dict["tmax"],
        tmin_sta=stns_dict["tmin"],
    )

    # precip second
    stns_dict = {}
    stn_vars = ["precip"]
    stns_dict["precip"] = adapter_factory(
        var=data_file,
        variable_name="precip",
        control=control,
    )

    # DEAL/CHECK missing values, the file has both -901.0 and -9999.0

    precip_forcings = PRMSStationPrecipForcing(
        control,
        discretization=discretization,
        parameters=parameters,
        precip_sta=stns_dict["precip"],
        tmaxf=temp_forcings.tmax,
        tminf=temp_forcings.tmin,
    )

    check_vars = stn_vars

    ans_file_var_name = {"precip": "prcp"}
    ans_file_var_comp_var = {"precip": "hru_ppt"}
    if do_compare_in_memory:
        answers = {}
        for var in check_vars:
            var_pth = simulation["dir"] / f"{ans_file_var_name[var]}.nc"
            answers[ans_file_var_comp_var[var]] = adapter_factory(
                var_pth, variable_name=ans_file_var_name[var], control=control
            )

    for ii in range(control.n_times):
        control.advance()
        temp_forcings.advance()
        precip_forcings.advance()
        temp_forcings.calculate(1.0)
        precip_forcings.calculate(1.0)

        if do_compare_in_memory:
            for var in answers.values():
                var.advance()
            # <
            compare_in_memory(
                precip_forcings,
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
