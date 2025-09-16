import pathlib as pl

import numpy as np
import pytest
import xarray as xr

import pywatershed as pws

# TODO build a subdomain of a test_domain for a check

# NHM domain section for some preliminary testing cheating (to be removed)
full_domain_dir = pl.Path(
    "/Users/jamesmcc/usgs/wu/science_base/irrigation_reanalysis/NHM"
)
full_control_file_nhm = full_domain_dir / "nhm_dynamic_2000_2020.control"
full_cbh_dir_nhm = full_domain_dir / "input"
full_cbh_nc_file_dict_nhm = {
    kk: full_cbh_dir_nhm / vv
    for kk, vv in {
        "precip_day": "prcp.nc",
        "tmax_day": "tmax.nc",
        "tmin_day": "tmin.nc",
    }.items()
}

use_nhm = False

# TO TEST:
# x PRMS inputs
# x [PRMS CBH files,  NetCDF CBH files]
# x [pywatershed outputs, PRMS outputs]
# x [known, single_seg]
# x [original index, shuffled index]

domain_name = "ucb_2yr"
domain_seg_outlet = {"ucb_2yr": np.int64(44425)}

cbh_types = [("ascii", "netcdf")[1:]]  # until implemented
output_types = [("pywatershed", "prms")[0:1]]  # until implemented

# compound test configs
subset_styles = ("known", "single_seg")[1:]
ind_order = ["orig", "shuffle"]
# id_seg_params = [("known", True), ("known", False), ("single_seg", None)][2:]
sub_ids_segs_types = []
for ss in subset_styles:
    for io in ind_order:
        # there is no way to shuffle single_seg subsetting
        if ss == "single_seg" and io == "shuffle":
            continue
        sub_ids_segs_types += [(ss, io)]


@pytest.fixture(scope="function")
def full_control_file(simulation):
    if simulation["name"] != f"{domain_name}:nhm":
        pytest.skip("test_domain_subset only runs for {domain_name}:nhm")

    if use_nhm:
        return full_control_file_nhm
    else:
        return simulation["control_file"]


@pytest.fixture(scope="function", params=cbh_types)
def full_cbh_nc_file_dict(simulation, request):
    cbh_type = request.param[0]
    if cbh_type == "netcdf":
        full_cbh_dir = simulation["control_file"].parent
        full_cbh_nc_file_dict = {
            kk: full_cbh_dir / vv
            for kk, vv in {
                "precip_day": "prcp.nc",
                "tmax_day": "tmax.nc",
                "tmin_day": "tmin.nc",
            }.items()
        }
    else:
        full_cbh_nc_file_dict = None

    if use_nhm:
        return full_cbh_nc_file_dict_nhm
    else:
        return full_cbh_nc_file_dict


@pytest.fixture(scope="function", params=sub_ids_segs_types)
def sub_ids_segs(simulation, request):
    subset_style = request.param[0]
    ind_order = request.param[1]

    if subset_style == "known":
        ctl = pws.Control.load_prms(
            simulation["control_file"], warn_unused_options=False
        )
        param_file = simulation["dir"] / ctl.options["parameter_file"]
        params = pws.parameters.PrmsParameters.load(param_file)
        nhm_ids = params.parameters["nhm_id"].copy()
        nhm_segs = params.parameters["nhm_seg"].copy()

        if ind_order == "shuffle":
            np.random.shuffle(nhm_ids)
            np.random.shuffle(nhm_segs)

        sub_ids_segs = {"nhm_ids": nhm_ids, "nhm_segs": nhm_segs}

    else:
        assert subset_style == "single_seg"
        sub_ids_segs = {
            "nhm_ids": None,
            "nhm_segs": np.int64(domain_seg_outlet[domain_name]),
        }

    return sub_ids_segs


@pytest.fixture(scope="function", params=output_types)
def output_format(simulation, request):
    return request.param[0]


def test_pws_subset_known_ids_segs(
    full_control_file,
    sub_ids_segs,
    full_cbh_nc_file_dict,
    output_format,
    tmp_path,
):
    subdomain = pws.utils.DomainSubset(
        full_control_file=full_control_file,
        sub_nhm_ids=sub_ids_segs["nhm_ids"],
        sub_nhm_segs=sub_ids_segs["nhm_segs"],
        full_cbh_nc_files_dict=full_cbh_nc_file_dict,
        from_seg_calc_check=True,
        output_format=output_format,
    )

    # can test that output_format=None raises error
    sub_domain_dir = tmp_path / "subdomain"
    subdomain.write(write_dir=sub_domain_dir, output_format="pywatershed")

    if output_format.lower() == "pywatershed":
        # test NHM
        nhm_proc_list = [
            pws.PRMSSolarGeometry,
            pws.PRMSAtmosphere,
            pws.PRMSCanopy,
            pws.PRMSSnow,
            pws.PRMSRunoff,
            pws.PRMSSoilzone,
            pws.PRMSGroundwater,
            pws.PRMSChannel,
        ]
        # full domain run
        full_dir = full_control_file.parent
        full_run_dir = tmp_path / "full_run"
        full_ctl = pws.Control.load_prms(
            full_control_file, warn_unused_options=False
        )
        full_param_file = full_dir / full_ctl.options["parameter_file"]
        full_params = pws.parameters.PrmsParameters.load(full_param_file)
        full_ctl.options["input_dir"] = full_dir
        full_ctl.options["netcdf_output_dir"] = pl.Path(full_run_dir)
        full_model = pws.Model(
            nhm_proc_list, control=full_ctl, parameters=full_params
        )
        full_model.run()
        # sub domain run
        sub_run_dir = tmp_path / "sub_run"
        sub_ctl_file = list(sub_domain_dir.glob("*.control"))[0]
        sub_ctl = pws.Control.load_prms(
            sub_ctl_file, warn_unused_options=False
        )
        sub_param_file = sub_domain_dir / sub_ctl.options["parameter_file"]
        sub_params = pws.parameters.PrmsParameters.from_netcdf(sub_param_file)
        sub_ctl.options["input_dir"] = sub_domain_dir
        sub_ctl.options["netcdf_output_dir"] = pl.Path(sub_run_dir)
        sub_model = pws.Model(
            nhm_proc_list, control=sub_ctl, parameters=sub_params
        )
        sub_model.run()

        nc_output_files = list(full_run_dir.glob("*.nc"))
        for nc_file in nc_output_files:
            if "budget" in nc_file.name:
                continue
            var_name = nc_file.with_suffix("").name
            file_name = f"{var_name}.nc"
            full_var = xr.open_dataset(full_run_dir / file_name).to_dataframe()
            sub_var = xr.open_dataset(sub_run_dir / file_name).to_dataframe()
            var_name = full_var.columns[0]
            mg = sub_var.join(
                full_var, how="left", lsuffix="_sub", rsuffix="_full"
            )
            if not (mg[f"{var_name}_sub"] == mg[f"{var_name}_full"]).all():
                raise ValueError(f"Comparison failed for {var_name=}")

    elif output_format.lower() == "prms":
        pass

    else:
        raise ValueError("How'd we get here?")
