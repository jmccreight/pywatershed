import numpy as np
import pytest

import pynhm.preprocess
from pynhm.atmosphere.NHMBoundaryLayer import NHMBoundaryLayer
from pynhm.budget.budget import Budget
from pynhm.canopy.PRMSCanopy import PRMSCanopy
from pynhm.utils.parameters import PrmsParameters

from utils import assert_or_print

start_time = np.datetime64("1979-01-03T00:00:00.00")
time_step = np.timedelta64(1, "D")

time_info = {
    "start_time": start_time,
    # "end_time": end_time,
    "time_step": time_step,
}


@pytest.fixture(scope="function")
def nhm_atm(domain):
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
@pytest.fixture(scope="function")
def nhm_canopy(domain, nhm_atm):
    # pkwater_equiv comes from snowpack; it is lagged by a time step
    prms_output_files = domain["prms_outputs"]
    fname = prms_output_files["pkwater_equiv"]
    df = pynhm.preprocess.CsvFile(fname).to_dataframe()
    pkwater_equiv = df.to_numpy()

    prms_params = PrmsParameters.load(domain["param_file"])

    cnp = PRMSCanopy(prms_params, nhm_atm, pkwater_equiv, **time_info)
    cnp.advance(itime_step=0)
    cnp.calculate(time_length=1.0)  # JLM suss
    cnp._current_time_index = 0  # JLM suss ++
    return cnp


def test_init_no_data():
    budget = Budget()
    assert len(budget._terms) == 0


def test_init_data(nhm_atm):
    data_dict = {"potet": nhm_atm}
    budget = Budget(data=data_dict)
    assert budget._terms == data_dict


def test_calculate(domain, nhm_atm, nhm_canopy):
    data_dict = {"potet": nhm_atm, "intcp_evap": nhm_canopy}
    budget = Budget(data=data_dict)
    budget.calculate()
    assert_or_print(
        {
            "calculate": budget.balance.mean() - 1
        },  # remove -1, this is just for canopy hack
        domain["test_ans"]["budget"],
        "budget",
        print_ans=domain["print_ans"],
    )

    # advance and check the second time?
