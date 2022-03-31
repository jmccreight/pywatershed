import numpy as np
import pytest

from pynhm.atmosphere.NHMBoundaryLayer import NHMBoundaryLayer
from pynhm.budget.budget import Budget

# from pynhm.canopy.PRMSCanopy import PRMSCanopy
from pynhm.utils.parameters import PrmsParameters
from utils import assert_or_print


@pytest.fixture(scope="function")
def nhm_atm(domain):
    start_time = np.datetime64("1979-01-03T00:00:00.00")
    time_step = np.timedelta64(1, "D")
    atm_information_dict = {
        "start_time": start_time,
        # "end_time": end_time,
        "time_step": time_step,
        "verbosity": 3,
        "height_m": 5,
    }
    var_translate = {
        "prcp_adj": "prcp",
        "rainfall_adj": "rainfall",
        "snowfall_adj": "snowfall",
        "tmax_adj": "tmax",
        "tmin_adj": "tmin",
        "swrad": "swrad",
        "potet": "potet",
    }
    var_file_dict = {
        var_translate[var]: file
        for var, file in domain["prms_outputs"].items()
        if var in var_translate.keys()
    }

    prms_params = PrmsParameters.load(domain["param_file"])

    atm = NHMBoundaryLayer.load_prms_output(
        **var_file_dict, parameters=prms_params, **atm_information_dict
    )
    atm.calculate_sw_rad_degree_day()
    atm.calculate_potential_et_jh()
    return atm


# fixture for canopy


def test_init_no_data():
    budget = Budget()
    assert len(budget._terms) == 0


def test_init_data(nhm_atm):
    data_dict = {"potet": nhm_atm}
    budget = Budget(data=data_dict)
    assert budget._terms == data_dict


def test_calculate(domain, nhm_atm):
    data_dict = {"potet": nhm_atm}
    budget = Budget(data=data_dict)
    budget.calculate()
    assert_or_print(
        {"calculate": budget.balance.mean()},
        domain["test_ans"]["budget"],
        "budget",
        print_ans=domain["print_ans"],
    )
