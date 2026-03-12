import pathlib as pl

import numpy as np
import pandas as pd
import pyPRMS as pp

import pywatershed as pws
from pywatershed.constants import pyprms_meta

new_param_names = ["spring_frost", "fall_frost"]

# The source data from the science_base release
# xr.open_dataset(fn)
wu_path = pl.Path("../../../wu/science_base/irrigation_reanalysis/NHM/input")
assert wu_path.exists()

# WU frost params
wu_param_file_names = [
    wu_path / "NHM.param",
    wu_path / "NHM_ag.param",
    wu_path / "frost_date_1980.param",
]
wu_param = pws.parameters.PrmsParameters.load(wu_param_file_names)
wu_sub_ds = wu_param.subset(new_param_names).to_xr_ds()

domain_names = ["drb_2yr", "hru_1", "ucb_2yr"]
for domain_name in domain_names:
    domain_dir = pl.Path(f"../{domain_name}/")

    param_file = domain_dir / "myparam.param"
    frost_param_file = domain_dir / "transp_frost.param"

    pfile = pp.ParameterFile(param_file, metadata=pyprms_meta, verbose=False)
    nhm_ids = pfile.parameters["nhm_id"].data
    wu_sub_data = wu_sub_ds.where(wu_sub_ds.nhm_id.isin(nhm_ids), drop=True)
    assert np.isin(wu_sub_data.nhm_id.values, nhm_ids).all()

    # need to re-order the data in some cases, use pandas DF.merge
    df_target = pd.DataFrame({"nhm_id": nhm_ids}).set_index("nhm_id")
    # print(df_target)
    wu_sub_data_df = (
        wu_sub_data.to_pandas()
        .reset_index()
        .drop(columns="nhru")
        .set_index("nhm_id")
    )
    # print(wu_sub_data_df)
    df_target = df_target.merge(
        wu_sub_data_df, left_index=True, right_index=True
    )
    # print(df_target)
    wu_sub_data = df_target.to_xarray()
    assert (wu_sub_data.nhm_id.values == nhm_ids).all()

    for vv in new_param_names:
        pfile.add(vv)
        pfile.parameters[vv].data = wu_sub_data[vv].values
        pfile.parameters[vv].dataw = wu_sub_data[vv].values

    all_params = list(pfile.parameters.keys())
    for vv in all_params:
        if vv in new_param_names:
            continue
        pfile.remove(vv)

    all_dims = list(pfile.dimensions.keys())
    for vv in all_dims:
        if vv == "nhru":
            continue
        pfile.dimensions.remove(vv)

    pfile.write_parameter_file(frost_param_file)
    assert frost_param_file.exists()

    # Remove the "header" of the file because this is like an "addendum"
    # to the full parameter file.
    frost_param_file_tmp = frost_param_file.with_suffix(".params_TMP")
    frost_param_file.rename(frost_param_file_tmp)
    with (
        open(frost_param_file_tmp, "r") as src,
        open(frost_param_file, "w") as target,
    ):
        skip = True
        for line in src:
            if "** Parameters **" in line:
                skip = False
                continue  # skip this line too
            if skip:
                continue
            target.write(line)
    frost_param_file_tmp.unlink()
