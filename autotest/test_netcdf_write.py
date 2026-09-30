import netCDF4 as nc4
import numpy as np
import pytest
import xarray as xr

from pywatershed.utils.netcdf_utils import NetCdfWrite


@pytest.mark.domainless
def test_string_coords_share_char_dim(tmp_path):
    # Issue 421: string coordinates are written as char arrays on a
    # "char<N>" dimension. The second string coordinate with the same max
    # length was skipped along with its dimension, so it was silently
    # missing from the file. Here node_maker_name and node_maker_id share
    # char12; node_maker_index is a plain integer coordinate.
    nnodes = 2
    extra_coords = {
        "node_coord": {
            "node_maker_name": np.array(["prms_channel", "starfit_node"]),
            "node_maker_index": np.array([0, 1]),
            "node_maker_id": np.array(["nhm_seg_0001", "starfit_4862"]),
        }
    }
    var_meta = {
        "node_outflows": {
            "dims": ["nnodes"],
            "type": "float64",
            "desc": "test",
            "units": "cfs",
        }
    }

    out_file = tmp_path / "node_outflows.nc"
    writer = NetCdfWrite(
        out_file,
        coordinates={"node_coord": np.arange(nnodes)},
        variables=["node_outflows"],
        var_meta=var_meta,
        extra_coords=extra_coords,
    )
    writer.close()

    # both string coordinates share the one char12 dimension
    with nc4.Dataset(out_file) as nc:
        assert nc.dimensions["char12"].size == 12
        assert nc["node_maker_name"].dimensions == ("node_coord", "char12")
        assert nc["node_maker_id"].dimensions == ("node_coord", "char12")

    # xarray folds the char dimension back into strings
    ds = xr.open_dataset(out_file, concat_characters=True)
    for coord_name, coord_data in extra_coords["node_coord"].items():
        assert coord_name in ds.variables
        assert ds[coord_name].values.tolist() == coord_data.tolist()
