import pathlib as pl
from textwrap import wrap
from typing import Callable, Tuple, Union

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from xyzservices import TileProvider

from ..base import meta
from ..base.model import Model
from ..base.process import Process
from ..utils.optional_import import import_optional_dependency


class ProcessPlot:
    def __init__(
        self,
        gis_dir: Union[str, pl.Path],
        hru_shp_file_name: str = "HRU_subset.shp",
        seg_shp_file_name: str = "Segments_subset.shp",
        hru_layer: str = None,
        seg_layer: str = None,
    ):
        """Plot process variables on HRU and segment geometries.

        Args:
            gis_dir: directory holding the GIS files.
            hru_shp_file_name: HRU polygons, a shapefile or a geodatabase
                (.gdb) directory; None for no HRU geometries.
            seg_shp_file_name: segment lines, as for hru_shp_file_name.
            hru_layer: layer name within hru_shp_file_name, needed when it
                is a geodatabase.
            seg_layer: as hru_layer, for seg_shp_file_name.
        """
        gpd = import_optional_dependency("geopandas")

        self.gis_dir = pl.Path(gis_dir)
        if hru_shp_file_name is not None:
            self.hru_shapefile = self.gis_dir / hru_shp_file_name
        else:
            self.hru_shapefile = None

        if seg_shp_file_name is not None:
            self.seg_shapefile = self.gis_dir / seg_shp_file_name
        else:
            self.seg_shapefile = None

        # HRU one-time setups
        if self.hru_shapefile is not None:
            self.hru_gdf = gpd.read_file(self.hru_shapefile, layer=hru_layer)

            # standardization manipulations based on a variety of different
            # conventions which have been found for the shp files
            if ("nhm_id" in self.hru_gdf.columns) and (
                "nhru_v1_1" in self.hru_gdf.columns
            ):
                # This is borderline non-sense
                self.hru_gdf = (
                    self.hru_gdf.drop("nhm_id", axis=1)
                    .rename(columns={"nhru_v1_1": "nhm_id"})
                    .set_index("nhm_id")
                )
            elif "GRID_CODE" in self.hru_gdf.columns:
                self.hru_gdf = self.hru_gdf.rename(
                    columns={"GRID_CODE": "nhm_id"}
                ).set_index("nhm_id")

            elif "GRIDCODE" in self.hru_gdf.columns:
                # e.g. the sagehen_mf6 geodatabase HRU layer, stored as float
                self.hru_gdf["nhm_id"] = self.hru_gdf["GRIDCODE"].astype(int)
                self.hru_gdf = self.hru_gdf.drop("GRIDCODE", axis=1).set_index(
                    "nhm_id"
                )

            else:
                msg = "Unidentified shp file convention, work needed"
                raise ValueError(msg)

        # segment one-time setup
        if self.seg_shapefile is not None:
            self.seg_gdf = gpd.read_file(self.seg_shapefile, layer=seg_layer)
            # if (self.__seg_poly.crs.name
            #     == "USA_Contiguous_Albers_Equal_Area_Conic_USGS_version"):
            #     print("Overriding USGS aea crs with EPSG:5070")
            self.seg_gdf.set_crs("EPSG:5070")

            self.seg_geoms_exploded = (
                self.seg_gdf.explode(index_parts=True)
                .reset_index(level=1, drop=True)
                .drop("model_idx", axis=1)
                .rename(columns={"nsegment_v": "nhm_seg"})
                .set_index("nhm_seg")
            )

        return

    def plot(self, var_name: str, process: Process, **kwargs):
        var_dims = list(meta.get_vars(var_name)[var_name]["dims"])
        if "nsegment" in var_dims:
            return self.plot_seg_var(var_name, process, **kwargs)
        elif "nhru" in var_dims:
            return self.plot_hru_var(var_name, process, **kwargs)
        else:
            raise ValueError()

    def plot_seg_var(
        self,
        var_name: str,
        process: Process,
        cmap: str = None,
        value_transform: Callable = None,
        figsize: tuple = (7, 10),
        title: str = None,
        aesthetic_width: bool = False,
        cx_map_source: TileProvider = cx.providers.CartoDB.Positron,
        vmin: float = None,
        vmax: float = None,
        aesthetic_width_color="darkblue",
        missing_kwds: dict = None,
    ):
        values = process[var_name]
        if value_transform is not None:
            values = value_transform(values)

        data_df = pd.DataFrame(
            {
                "nhm_seg": process._params.coords["nhm_seg"],
                var_name: values,
            }
        ).set_index("nhm_seg")
        df_plot = self.seg_geoms_exploded.join(data_df).reset_index()
        if aesthetic_width:
            ax = df_plot.plot(
                column=var_name,
                figsize=figsize,
                linewidth=df_plot[var_name],
                edgecolor=aesthetic_width_color,
            )
        else:
            if vmin is None:
                vmin = np.nanmin(values)
            if vmax is None:
                vmax = np.nanmax(values)
            if cmap is None:
                cmap = "cool"
            if missing_kwds is None:
                missing_kwds = {
                    "color": "lightgrey",
                    "linewidth": 0.5,
                }

            ax = df_plot.plot(
                column=var_name,
                figsize=figsize,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                legend=True,
                missing_kwds=missing_kwds,
            )

        cx.add_basemap(
            ax=ax,
            crs=df_plot.crs,
            source=cx_map_source,
        )
        ax.set_axis_off()
        if title is None:
            title = var_name
        _ = ax.set_title(title)

        plt.show()
        return

    def get_hru_var(self, var_name: str, model: Model):
        # find the process
        for proc_name, proc in model.processes.items():
            params_vars = list(set(proc.variables) | set(proc.parameters))
            if var_name in params_vars:
                process = proc
                break

        data_df = pd.DataFrame(
            {
                "nhm_id": process._params.coords["nhm_id"],
                var_name: process[var_name],
            }
        ).set_index("nhm_id")
        return data_df

    def plot_hru_var(
        self,
        var_name: str,
        process: Process,
        data: np.ndarray = None,
        data_units: str = None,
        nhm_id: np.ndarray = None,
        clim: Tuple[float] = None,
        time: np.ndarray = None,
        **kwargs,
    ):
        """Map a variable on the HRU polygons.

        Args:
            var_name: the variable name, used for the title and colorbar.
            process: the process holding the variable; its current values
                are mapped when data is None.
            data: values to map instead of the process's current state,
                either (nhru,) for a single map or (ntime, nhru) for a map
                per time (rendered with a time widget, see ``time``).
            data_units: units of data, used when var_name has no metadata.
            nhm_id: the HRU ids matching data; required with data.
            clim: colorbar limits; with a time dimension these are fixed
                across all times (otherwise each time scales separately).
            time: the times for 2-d data; may be omitted when data is an
                xarray.DataArray with a "time" coordinate.
            **kwargs: passed to hvplot, overriding the defaults set here.
        """
        _ = import_optional_dependency("hvplot.pandas")

        ccrs = import_optional_dependency("cartopy.crs")

        if data is None:
            # data_df = self.get_hru_var(var_name, model)
            data_df = pd.DataFrame(
                {
                    "nhm_id": process._params.coords["nhm_id"],
                    var_name: process[var_name],
                }
            ).set_index("nhm_id")

        else:
            if nhm_id is None:
                # nhm_id = model.parameters["nhm_id"]
                raise ValueError("code needs work to handle nhm_id=None")

            is_dataarray = hasattr(data, "coords")
            if time is None and is_dataarray and "time" in data.coords:
                time = data["time"].values

            if np.ndim(data) == 2:
                if time is None:
                    raise ValueError("2-d data needs time")
                data_df = pd.DataFrame(
                    np.asarray(data),
                    index=pd.Index(pd.to_datetime(time), name="time"),
                    columns=pd.Index(np.asarray(nhm_id), name="nhm_id"),
                )
                # long table: one row per (time, nhm_id)
                data_df = data_df.melt(
                    ignore_index=False, value_name=var_name
                ).reset_index()
            else:
                data_df = pd.DataFrame(
                    {
                        "nhm_id": nhm_id,
                        var_name: data,
                    }
                ).set_index("nhm_id")

        if "time" in data_df.columns:
            plot_df = self.hru_gdf.merge(
                data_df, left_index=True, right_on="nhm_id"
            )
        else:
            plot_df = self.hru_gdf.join(data_df)

        metadata = meta.get_vars(var_name)
        if not len(metadata):
            metadata = meta.get_params(var_name)
        if len(metadata):
            metadata = metadata[var_name]
        else:
            metadata = None

        # hvplot needs the CRS the geometries are in. Use the layer's own
        # when it declares one; the drb shapefiles do not and are EPSG:5070.
        epsg = None
        if plot_df.crs is not None:
            epsg = plot_df.crs.to_epsg()
        if epsg is None:
            epsg = 5070

        # Size the frame to the domain's aspect ratio (the tiled map forces
        # equal axes, so a fixed height alone crops wide domains east-west).
        # The tiles draw in Web Mercator (EPSG:3857), so measure the aspect
        # there, not in the layer's own projection.
        mercator_bounds = (
            plot_df.set_crs(epsg, allow_override=True)
            .to_crs(3857)
            .total_bounds
        )
        xmin, ymin, xmax, ymax = mercator_bounds
        aspect = (xmax - xmin) / (ymax - ymin)
        # Initial view: the whole domain with a small margin (the default
        # auto-range with tiles can leave part of it out of view). These
        # are applied to the finished plot, in Mercator, rather than passed
        # to hvplot: hvplot projects limits through cartopy, whose UTM
        # projections clip to the zone's nominal longitude range, which
        # shifts the view for domains near a zone edge (e.g. sagehen, just
        # west of zone 11).
        margin = 0.05
        xpad = margin * (xmax - xmin)
        ypad = margin * (ymax - ymin)
        xlim = (xmin - xpad, xmax + xpad)
        ylim = (ymin - ypad, ymax + ypad)
        frame_height = 550
        frame_width = int(frame_height * aspect)
        max_width = 900
        if frame_width > max_width:
            frame_width = max_width
            frame_height = int(max_width / aspect)

        title = f'"{var_name}"\n'
        clabel = data_units
        if metadata is not None:
            title += "\n".join(
                wrap(
                    f"{metadata['desc']}, {metadata['units']}",
                    width=frame_width / 10,
                )
            )
            clabel = f"{metadata['units']}"

        args = {
            "tiles": True,
            "crs": ccrs.epsg(epsg),
            "frame_height": frame_height,
            "frame_width": frame_width,
            "c": var_name,
            "line_width": 0,
            "alpha": 0.75,
            "hover_cols": ["nhm_id"],
            "title": title,
            "clabel": clabel,
            "xlabel": "Longitude (degrees East)",
            "ylabel": "Latitude (degrees North)",
        } | kwargs

        if clim is not None:
            args["clim"] = clim

        if "time" in plot_df.columns:
            # One map per time, with a time widget. All frames are built up
            # front (dynamic=False): on-demand frames from two widgets
            # playing at once race and show each other's data. The
            # {dimensions} placeholder puts the current time in the title.
            args = {"groupby": "time", "dynamic": False} | args
            args["title"] = args["title"] + "\n{dimensions}"

        plot = plot_df.hvplot(**args)
        if args.get("tiles", True):
            plot = plot.opts(xlim=xlim, ylim=ylim)
        return plot


def plot_line_collection(
    ax,
    geoms,
    values=None,
    cmap=None,
    norm=None,
    vary_width=False,
    vary_color=True,
    colors=None,
    alpha=1.0,
    linewidth=1.0,
    **kwargs,
):
    """Plot a collection of line geometries"""
    shapely = import_optional_dependency("shapely")

    lines = []
    for geom in geoms:
        a = np.asarray(geom.coords)

        if geom.has_z:
            a = shapely.geometry.LineString(zip(*geom.xy))

        lines.append(shapely.geometry.LineString(a))

    if vary_width:
        lwidths = ((values / values.max()).to_numpy() + 0.01) * linewidth
        if vary_color:
            lines = LineCollection(
                lines, linewidths=lwidths, cmap=cmap, norm=norm, alpha=alpha
            )
        else:
            lines = LineCollection(
                lines, linewidths=lwidths, colors=colors, alpha=alpha
            )
    elif vary_color:
        lines = LineCollection(
            lines, linewidth=linewidth, alpha=alpha, cmap=cmap, norm=norm
        )

    if vary_color and values is not None:
        lines.set_array(values)
        # lines.set_cmap(cmap)

    ax.add_collection(lines, autolim=True)
    ax.autoscale_view()
    return lines


def plot_polygon_collection(
    ax,
    geoms,
    values=None,
    cmap=None,
    norm=None,
    facecolor=None,
    edgecolor=None,
    alpha=1.0,
    linewidth=1.0,
    **kwargs,
):
    """Plot a collection of Polygon geometries"""
    # from https://stackoverflow.com/questions/33714050/
    #     geopandas-plotting-any-way-to-speed-things-up
    shapely = import_optional_dependency("shapely")

    patches = []

    for poly in geoms:
        a = np.asarray(poly.exterior)
        if poly.has_z:
            a = shapely.geometry.Polygon(zip(*poly.exterior.xy))

        patches.append(Polygon(a))

    patches = PatchCollection(
        patches,
        facecolor=facecolor,
        linewidth=linewidth,
        edgecolor=edgecolor,
        alpha=alpha,
        cmap=cmap,
        norm=norm,
    )

    if values is not None:
        patches.set_array(values)
        # patches.set_cmap(cmap)

    ax.add_collection(patches, autolim=True)
    ax.autoscale_view()
    return patches
