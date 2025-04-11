import numpy as np
import pytest
import xarray as xr

import pywatershed as pws

process_cases = [
    [pws.PRMSSolarGeometry, pws.PRMSAtmosphere],
    [pws.PRMSSoilzone, pws.PRMSGroundwater, pws.PRMSChannel],
]


def get_control(file):
    control = pws.Control.load_prms(file, warn_unused_options=False)
    control.options["verbosity"] = 10
    control.options["budget_type"] = None
    control.options["calc_method"] = "numba"
    control.options["netcdf_output_dir"] = None
    control.edit_end_time(control.start_time + 14 * control.time_step)
    return control


def get_params(control, sim_dir):
    return pws.parameters.PrmsParameters.load(
        sim_dir / control.options["parameter_file"]
    )


# use this ficture to skip non-nhm tests
@pytest.fixture(scope="function")
def control(simulation):
    sim_name = simulation["name"]
    config_name = sim_name.split(":")[1]
    if config_name != "nhm":
        pytest.skip(
            "The configuration is not tested by test_model_inputs: "
            f"{config_name}"
        )
    control = get_control(simulation["control_file"])
    return control


@pytest.mark.parametrize("processes", [process_cases[0]])
def test_cbh_input_paths(simulation, control, processes, tmp_path):
    c0 = get_control(simulation["control_file"])
    p0 = get_params(c0, simulation["dir"])
    with pytest.raises(ValueError):
        _ = pws.Model(processes, control=c0, parameters=p0)
    del c0, p0

    c1 = get_control(simulation["control_file"])
    c1.options["input_dir"] = "foo"
    c1.options["input_file"] = "bar"
    p1 = get_params(c1, simulation["dir"])
    with pytest.raises(ValueError):
        _ = pws.Model(processes, control=c1, parameters=p1)
    del c1, p1

    # use existing, separate input files
    c2 = get_control(simulation["control_file"])
    if pws.PRMSAtmosphere in processes:
        c2.options["input_dir"] = simulation["dir"]
    else:
        c2.options["input_dir"] = simulation["output_dir"]
    p2 = get_params(c2, simulation["dir"])
    m2 = pws.Model(processes, control=c2, parameters=p2)
    m2.run(finalize=True)

    # generate a combined input file like cbh.nc
    # The input file names are a private attribute of the model.
    input_file = tmp_path / "input_file.nc"
    input_dir = c2.options["input_dir"]
    input_file_vars = m2._file_inputs
    input_da_list = []
    for vv in input_file_vars:
        input_da_list += [xr.open_dataarray(input_dir / f"{vv}.nc")]

    _ = xr.merge(input_da_list).to_netcdf(input_file)

    c3 = get_control(simulation["control_file"])
    c3.options["input_file"] = input_file
    p3 = get_params(c3, simulation["dir"])
    m3 = pws.Model(processes, control=c3, parameters=p3)
    m3.run(finalize=True)

    for pp in m3.processes.keys():
        for vv in m3.processes[pp].variables:
            data3 = m3.processes[pp][vv]
            data2 = m2.processes[pp][vv]
            if isinstance(data3, pws.TimeseriesArray):
                np.testing.assert_equal(data3.current, data2.current)
            else:
                np.testing.assert_equal(data3, data2)
