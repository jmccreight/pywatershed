"""Make STARFIT parameters file

TODO: Retain multiple reservoirs in-series

STARFIT data and inspiration taken from
link:[https://code.usgs.gov/wma/pywatershed++_++reservoirs/starfit-pywatershed]

Overview:

    Given the domain:
    * Get all the starfit reservoirs in the domain
    * Remove any serially linked (multiple between nhm segments)
    * Visually inspect with a plot.
    * Manually edit (e.g. remove mistakes from the crosswalk).
    * Calculate a climatological initial storage for day of year by
    averaging that day from available resops data.
    * Write out STARFIT netcdf parameter file
"""

from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import pywatershed as pws
from pywatershed.utils import import_optional_dependency

folium = import_optional_dependency("folium")


class MakeStarfitParams:
    """Interactively create Starfit parameter file from various inputs"""

    def __init__(
        self,
        control_file,
        grand_file,
        grand_nhm_seg_crosswalk_file,
        starfit_param_csv_file,
        resops_inflow_file,
        resops_outflow_file,
        resops_storage_file,
        seg_shp_file: str = None,
        hru_shp_file: str = None,
        prms_gpkg_file: str = None,
    ) -> None:
        if (not (seg_shp_file and hru_shp_file) and not (prms_gpkg_file)) or (
            (seg_shp_file and hru_shp_file) and prms_gpkg_file
        ):
            msg = (
                "Either both of seg_shp_file and hru_shp file are required OR "
                "prms_gpkg_file is required (but not both)."
            )
            raise ValueError(msg)

        # inputs
        self._control_file = control_file
        self._grand_file = grand_file
        self._grand_crosswalk_file = grand_nhm_seg_crosswalk_file
        self._seg_shp_file = seg_shp_file
        self._hru_shp_file = hru_shp_file
        self._prms_gpkg_file = prms_gpkg_file
        self._starfit_param_csv_file = starfit_param_csv_file
        self._resops_inflow_file = resops_inflow_file
        self._resops_outflow_file = resops_outflow_file
        self._resops_storage_file = resops_storage_file

        self._pws_ingest()
        self._crosswalk_ingest()
        self._grand_file_ingest()
        self._shp_files_ingest()
        self._istarf_ingest()
        self._resops_ingest()

        self._intersect_grand_segs()

    @property
    def dataset(self):
        return self._ds_local_starfit_params

    def _pws_ingest(self):
        # PWS control
        control_file_dir = self._control_file.parent
        self._control = pws.Control.load_prms(self._control_file)
        self._start_time = self._control.start_time
        self._end_time = self._control.end_time

        # PWS parameters
        self._parameter_file = (
            control_file_dir / self._control.options["parameter_file"]
        )
        self._parameters = pws.parameters.PrmsParameters.load(
            self._parameter_file
        )

    def _crosswalk_ingest(self):
        # import GRanD crosswalk
        self._pd_grand_crosswalk = pd.read_csv(self._grand_crosswalk_file)
        self._pd_grand_crosswalk.onoffnet = (
            self._pd_grand_crosswalk.onoffnet.to_numpy(dtype="int64")
        )
        self._pd_grand_crosswalk.gfv11_id = (
            self._pd_grand_crosswalk.gfv11_id.to_numpy(dtype="int64")
        )

    def _grand_file_ingest(self) -> None:
        self._gpd_grand = gpd.read_file(self._grand_file)

    def _shp_files_ingest(self):
        if self._prms_gpkg_file:
            self._gpd_seg = gpd.read_file(
                self._prms_gpkg_file, layer="nsegment"
            )
            self._gpd_hru = gpd.read_file(self._prms_gpkg_file, layer="nhru")
        else:
            self._gpd_seg = gpd.read_file(self._seg_shp_file)
            self._gpd_hru = gpd.read_file(self._hru_shp_file)

    def _istarf_ingest(self):
        self._pd_conus_starfit_params = pd.read_csv(
            self._starfit_param_csv_file
        )

    def _resops_ingest(self):
        dtype_dict = defaultdict(lambda: np.float64, {"date": str})
        pd_resops_inflow = pd.read_csv(
            self._resops_inflow_file, index_col="date", dtype=dtype_dict
        )
        pd_resops_inflow.index = pd_resops_inflow.index.astype(
            "datetime64[ns]"
        ).rename("time")

        pd_resops_outflow = pd.read_csv(self._resops_outflow_file).set_index(
            "date"
        )
        pd_resops_outflow.index = pd_resops_outflow.index.astype(
            "datetime64[ns]"
        ).rename("time")

        pd_resops_storage = pd.read_csv(self._resops_storage_file).set_index(
            "date"
        )
        pd_resops_storage.index = pd_resops_storage.index.astype(
            "datetime64[ns]"
        ).rename("time")

        # convert flow from cubic meters to cubic feet, all are per second
        pd_resops_inflow *= pws.constants.cm_to_cf
        pd_resops_outflow *= pws.constants.cm_to_cf

        self.pd_resops_inflow = pd_resops_inflow
        self.pd_resops_outflow = pd_resops_outflow
        self.pd_resops_storage = pd_resops_storage

    def _intersect_grand_segs(self, grand_ids_drop: list = None):
        # get list of the grand_id of reservoirs in the model the first time
        # Skip this on subsequent calls, using what is already found/dropped.
        if not hasattr(self, "_list_grand_in_model"):
            self._list_grand_in_model = self._pd_grand_crosswalk[
                self._pd_grand_crosswalk.gfv11_id.isin(
                    self._parameters.parameters["nhm_seg"]
                )
            ].grand_id

        if grand_ids_drop:
            self._list_grand_in_model = self._list_grand_in_model[
                ~self._list_grand_in_model.isin(grand_ids_drop)
            ]

        # get list of the segs coinciding with grand reservoirs in the model
        self._list_segs_with_grand = self._pd_grand_crosswalk[
            self._pd_grand_crosswalk.grand_id.isin(self._list_grand_in_model)
        ].gfv11_id
        # the crosswalk: xw
        self._xw = self._pd_grand_crosswalk[["grand_id", "gfv11_id"]].rename(
            columns={"grand_id": "GRAND_ID"}
        )
        # get gis spatial info
        gpd_grand_selected = self._gpd_grand[
            self._gpd_grand.GRAND_ID.isin(self._list_grand_in_model)
        ]
        gpd_grand_selected = gpd_grand_selected.to_crs(self._gpd_seg.crs)
        gpd_grand_selected = gpd_grand_selected.merge(
            self._xw, on="GRAND_ID", how="inner"
        )
        gpd_grand_selected["add_to_nhm"] = gpd_grand_selected.GRAND_ID.isin(
            self._list_grand_in_model
        )
        self._gpd_grand_selected = gpd_grand_selected

        self._nreservoirs_per_seg()
        self._merge_istarf_params()
        self._merge_resops_params()

    def plot_grand_in_domain(
        self,
        fields=["GRAND_ID", "RES_NAME", "DAM_NAME", "add_to_nhm", "gfv11_id"],
        colors=["darkblue", "pink"],
        radius=1400,
        fill_color="pink",
        fill_opacity=0.9,
        color="black",
        weight=1,
    ):
        # great example here
        # https://python-visualization.github.io/folium/latest/user_guide/geojson/geojson_marker.html#Use-a-Circle-as-a-Marker  # noqa: E501
        starfit_layer = folium.GeoJson(
            self._gpd_grand_selected,
            name="STARFIT",
            control=True,
            marker=folium.Circle(
                radius=radius,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                color=color,
                weight=weight,
            ),
            tooltip=folium.GeoJsonTooltip(fields=fields),
            popup=folium.GeoJsonPopup(fields=fields),
            style_function=lambda x: {
                "fillColor": colors[x["properties"]["add_to_nhm"]],
            },
            highlight_function=lambda x: {"fillOpacity": 0.1},
        )

        _ = pws.plot.DomainPlot(
            hru_shp_file=self._gpd_hru,
            segment_shp_file=self._gpd_seg,
            more_layers=starfit_layer,
        )

    def drop_grand_ids(self, grand_ids_drop: list) -> None:
        """Drop grand IDs from the domain.

        Args:
            grand_ids_drop: A list of the grand_ids to drop.
        """
        self._intersect_grand_segs(grand_ids_drop)

    def _nreservoirs_per_seg(self) -> None:
        # some segments have more than one istarf reservoir, but flowgraph
        # currently can only have one find which are duplicated
        (ar, index, inverse, counts) = np.unique(
            self._list_segs_with_grand,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        segs_with_multiple_grand = ar[np.where(counts > 1)[0]]

        # Find out which duplicated grand to remove
        # get indices of the duplicated grand reservoirs not greatest reservoir
        # # volume on that segment
        self._small_dups_to_rm = []
        for seg in segs_with_multiple_grand:
            repeat_grand = self._pd_grand_crosswalk[
                self._pd_grand_crosswalk.gfv11_id == seg
            ].grand_id
            repeat_grand_max_CAP_idx = np.argmax(
                self._gpd_grand[
                    self._gpd_grand.GRAND_ID.isin(repeat_grand)
                ].CAP_MCM
            )

            # Could print diagnostics for different scenarios

            # remove these grand_ids from param file
            grand_remove = repeat_grand[
                repeat_grand.index
                != repeat_grand.index[repeat_grand_max_CAP_idx]
            ]
            print(
                f"\nSegment {seg} has multiple grand reservoirs:\n"
                f"{list(repeat_grand)}.\n"
                f"Keeping largest capacity reservoir on {seg} would remove "
                f"grand_ids: {list(grand_remove)}"
            )
            self._small_dups_to_rm += list(grand_remove)

    def rm_small_duplicates(self):
        self._intersect_grand_segs(self._small_dups_to_rm)

    def _merge_istarf_params(self):
        # keep only the selected reservoirs that meet criteria (in the model,
        # not duplicated on segments)
        pd_local_starfit_params = self._pd_conus_starfit_params[
            self._pd_conus_starfit_params["GRanD_ID"].isin(
                self._list_grand_in_model
            )
        ]
        # fix character values
        pd_local_starfit_params = pd_local_starfit_params.replace(
            "#NAME?", -np.inf
        )
        pd_local_starfit_params.NORhi_min = (
            pd_local_starfit_params.NORhi_min.to_numpy(dtype="float64")
        )
        pd_local_starfit_params.NORlo_min = (
            pd_local_starfit_params.NORlo_min.to_numpy(dtype="float64")
        )
        pd_local_starfit_params = (
            pd_local_starfit_params.merge(
                self._xw.rename(columns={"GRAND_ID": "GRanD_ID"}),
                on="GRanD_ID",
                how="inner",
            )
            .rename(columns={"gfv11_id": "nhm_seg"})
            .set_index("nhm_seg")
        )
        # convert pd to ds
        ds_local_starfit_params = (
            pd_local_starfit_params.to_xarray().rename_dims(
                {"nhm_seg": "nreservoirs"}
            )
        )
        # this embeds the crosswalk into the parameter file
        ds_local_starfit_params["nhm_seg"].attrs = {
            "desc": "nhm_seg to which nodes flow"
        }
        ds_local_starfit_params["grand_id"] = ds_local_starfit_params[
            "GRanD_ID"
        ]
        self._ds_local_starfit_params = ds_local_starfit_params

    def _merge_resops_params(self):
        # assign ResOpsUS storage at start time as initial_storage parameter in
        # starfit_params
        pd_resops_storage_init = []
        for grand in self._ds_local_starfit_params.GRanD_ID.data:
            grand = str(grand)
            if grand in self.pd_resops_storage.columns:
                min_yr = self.pd_resops_storage[grand].index.min().year
                max_yr = self.pd_resops_storage[grand].index.max().year
                start_md = self._start_time.item().strftime("%m-%d")
                start_doy_list = []

                for yy in range(min_yr, max_yr + 1):
                    date_try = f"{yy}-{start_md}"
                    if date_try in self.pd_resops_storage[grand].index:
                        start_doy_list.append(date_try)

                mean_storage_doy = self.pd_resops_storage[grand][
                    start_doy_list
                ].mean(skipna=True)
                pd_resops_storage_init.append(
                    mean_storage_doy * pws.constants.cm_to_cf
                )  # convert from MCM to cf
            else:
                pd_resops_storage_init.append(np.nan)

        # get initial storage data from ResOps
        self._ds_local_starfit_params["initial_storage"] = xr.Variable(
            "nreservoirs",
            pd_resops_storage_init,
        )

        # add missing variables from the csv file to the dataset
        self._ds_local_starfit_params["initial_storage"] = xr.Variable(
            "nreservoirs",
            np.zeros(
                [self._ds_local_starfit_params.sizes["nreservoirs"]],
                dtype="float64",
            )
            * np.nan,
        )

        self._ds_local_starfit_params["inflow_mean"] = xr.Variable(
            "nreservoirs",
            np.ones(
                [self._ds_local_starfit_params.sizes["nreservoirs"]],
                dtype="float64",
            )
            * 10,
        )

        self._ds_local_starfit_params["start_time"] = xr.Variable(
            "nreservoirs",
            np.array(
                ["NaT"] * self._ds_local_starfit_params.sizes["nreservoirs"],
                dtype="datetime64[ns]",
            ),
        )
        self._ds_local_starfit_params["end_time"] = (
            self._ds_local_starfit_params["start_time"]
        )

    def to_netcdf(self, starfit_param_file_name):
        self._ds_local_starfit_params.to_netcdf(starfit_param_file_name)
