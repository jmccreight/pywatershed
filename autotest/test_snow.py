import numpy as np
import pytest

from pynhm.snow.NHMSnow import NHMSnow
from pynhm.utils import ControlVariables

snow_var_keys = ["pkwater_equiv", "snowcov_area", "snowmelt"]

control_keys = ["start_time", "end_time", "initial_deltat"]


@pytest.fixture(scope="function")
def control(domain):
    control_file = domain["control_file"]
    control = ControlVariables.load(control_file)
    control = control[control_keys]
    control["time_step"] = control["initial_deltat"]
    del control["initial_deltat"]
    return control


@pytest.fixture(scope="function")
def snow_nhm_init_prms_output(control, domain):
    var_file_dict = {
        key: file
        for key, file in domain["prms_outputs"].items()
        if key in snow_var_keys
    }
    snow = NHMSnow.load_prms_output(**var_file_dict, **control)
    return snow


def test_init_prms_output(snow_nhm_init_prms_output):
    # Tests thats the snow_nhm_init_prms_output fixture is properly created.
    # Could also check values but that will be redundant when checking
    # pynhm solutions.
    assert sorted(snow_nhm_init_prms_output.variables) == sorted(snow_var_keys)
    return
