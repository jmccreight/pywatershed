import pathlib as pl

import numpy as np
import pytest

import pywatershed as pws

# TODO build a subdomain of a test_domain for a check
# TODO test a perturbation of the nhm_id order wrt the full domain

domain_name = "ucb_2yr"
domain_seg_outlet = {"ucb_2yr": np.int64(44425)}

full_domain_dir = pl.Path(
    "/Users/jamesmcc/usgs/wu/science_base/irrigation_reanalysis/NHM"
)
full_cbh_dir_nhm = full_domain_dir / "input"
full_cbh_file_list_nhm = [
    full_cbh_dir_nhm / ff for ff in ["prcp.nc", "tmax.nc", "tmin.nc"]
]

full_control_file_nhm = full_domain_dir / "nhm_dynamic_2000_2020.control"

subset_style = ("known", "single_seg")
shuffle_ids = [False, True]
# there is no way to shuffle single_seg subsetting
id_seg_params = [("known", True), ("known", False), ("single_seg", None)][2:]


@pytest.fixture(scope="function")
def full_control_file(simulation):
    if simulation["name"] != f"{domain_name}:nhm":
        pytest.skip("test_domain_subset only runs for {domain_name}:nhm")

    # return full_control_file_nhm
    return simulation["control_file"]


@pytest.fixture(scope="function")
def full_cbh_file_list(simulation):
    full_cbh_dir = simulation["control_file"].parent
    full_cbh_file_list = [
        full_cbh_dir / ff for ff in ["prcp.nc", "tmax.nc", "tmin.nc"]
    ]
    # return full_cbh_file_list_nhm
    return full_cbh_file_list


@pytest.fixture(scope="function", params=id_seg_params)
def sub_ids_segs(simulation, request):
    sub_style = request.param[0]
    shuffle = request.param[1]
    if sub_style == "known":
        ctl = pws.Control.load_prms(
            simulation["control_file"], warn_unused_options=False
        )
        param_file = simulation["dir"] / ctl.options["parameter_file"]
        params = pws.parameters.PrmsParameters.load(param_file)
        nhm_ids = params.parameters["nhm_id"].copy()
        nhm_segs = params.parameters["nhm_seg"].copy()

        if shuffle:
            np.random.shuffle(nhm_ids)
            np.random.shuffle(nhm_segs)

        sub_ids_segs = {"nhm_ids": nhm_ids, "nhm_segs": nhm_segs}

    else:
        sub_ids_segs = {
            "nhm_ids": None,
            "nhm_segs": np.int64(domain_seg_outlet[domain_name]),
        }

    return sub_ids_segs


# TODO;
# test both known ids+segs and a desired outlet
# test run with PRMS and with pws separately (but not PRMS w pws)


def test_pws_subset_known_ids_segs(
    full_control_file, sub_ids_segs, full_cbh_file_list
):
    subdomain = pws.utils.DomainSubset(
        full_cbh_files_list=full_cbh_file_list,
        full_control_file=full_control_file,
        sub_nhm_ids=sub_ids_segs["nhm_ids"],
        sub_nhm_segs=sub_ids_segs["nhm_segs"],
        output_format="pywatershed",
        from_seg_calc_check=True,
    )
