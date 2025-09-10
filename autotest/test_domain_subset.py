import pathlib as pl

import numpy as np
import pytest

import pywatershed as pws

# TODO build a subdomain of a test_domain for a check

domain_name = "ucb_2yr"
domain_seg_outlet = {"ucb_2yr": np.int64(44425)}

full_domain_dir = pl.Path(
    "/Users/jamesmcc/usgs/wu/science_base/irrigation_reanalysis/NHM"
)
full_cbh_dir = full_domain_dir / "input"
full_control_file_nhm = full_domain_dir / "nhm_dynamic_2000_2020.control"

subset_style = ("known", "single_seg")[0:1]


@pytest.fixture(scope="function")
def full_control_file(simulation):
    if simulation["name"] != f"{domain_name}:nhm":
        pytest.skip("test_domain_subset only runs for {domain_name}:nhm")

    # TODO: change to subdomain of one of the test_domains
    # return simulation["control_file"]
    return full_control_file_nhm


@pytest.fixture(scope="function", params=subset_style)
def sub_ids_segs(simulation, request):
    if request.param == "known":
        ctl = pws.Control.load_prms(
            simulation["control_file"], warn_unused_options=False
        )
        param_file = simulation["dir"] / ctl.options["parameter_file"]
        params = pws.parameters.PrmsParameters.load(param_file)
        sub_ids_segs = {
            "nhm_ids": params.parameters["nhm_id"],
            "nhm_segs": params.parameters["nhm_seg"],
        }
    else:
        sub_ids_segs = {
            "nhm_ids": None,
            "nhm_segs": np.int64(domain_seg_outlet[domain_name]),
        }

    return sub_ids_segs


# TODO;
# test both known ids+segs and a desired outlet
# test run with PRMS and with pws separately (but not PRMS w pws)


def test_pws_subset_known_ids_segs(full_control_file, sub_ids_segs):
    subdomain = pws.utils.DomainSubset(
        output_format="pywatershed",
        full_cbh_dir=full_cbh_dir,
        full_control_file=full_control_file,
        sub_ids=sub_ids_segs["nhm_ids"],
        sub_segs=sub_ids_segs["nhm_segs"],
    )
