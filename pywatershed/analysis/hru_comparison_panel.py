#!/usr/bin/env python
# coding: utf-8

"""
HRU Comparer - Interactive comparison of model runs by HRU

This module provides a flexible class for comparing multiple model runs
with interactive visualization of spatial patterns and timeseries.
"""

import pathlib as pl
from typing import Dict, List, Optional, Union

import geopandas as gpd
import geoviews as gv
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from cartopy import crs as ccrs
from holoviews import streams


class HRUComparisonPanel:
    """
    Interactive comparison tool for HRU-based model outputs.

    This class provides an interactive Panel-based interface for comparing
    multiple model runs across different variables, with spatial visualization
    and timeseries plotting capabilities.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to shapefile containing HRU polygons
    variable_names : list of str
        List of variable names to compare (should match netcdf filenames without .nc)
    run_directories : dict
        Dictionary mapping run names to directory paths containing netcdf files
        Example: {"Run1": "/path/to/run1/output", "Run2": "/path/to/run2/output"}
    hru_id_column : str, optional
        Column name in shapefile containing HRU IDs. If None, will auto-detect.
    map_width : int, optional
        Width of map plot in pixels (default: 1200)
    map_height : int, optional
        Height of map plot in pixels (default: 650)
    timeseries_width : int, optional
        Width of timeseries plots in pixels (default: 1400)
    timeseries_height : int, optional
        Height of each timeseries plot in pixels (default: 250)
    colormap : str, optional
        Default colormap to use for spatial plots (default: "viridis").
        Can be changed interactively in the app.
    simplify_tolerance : int, optional
        Tolerance in meters for geometry simplification (default: 100).
        Increase this (e.g., 500 or 1000) for faster rendering with large domains.

    Examples
    --------
    >>> comparer = HRUComparisonPanel(
    ...     shapefile_path="model_nhru.shp",
    ...     variable_names=["prcp", "tmean", "hru_ppt"],
    ...     run_directories={
    ...         "Republican": "/path/to/republican/output",
    ...         "NHM": "/path/to/nhm/output",
    ...     },
    ... )
    >>> app = comparer.create_app()
    >>> app.show()  # Display in notebook
    >>> # or app.servable() for deployment
    """

    def __init__(
        self,
        shapefile_path: Union[str, pl.Path],
        variable_names: List[str],
        run_directories: Dict[str, Union[str, pl.Path]],
        hru_id_column: Optional[str] = "nhru_v1_1",
        map_width: int = 1200,
        map_height: int = 650,
        timeseries_width: int = 1400,
        timeseries_height: int = 250,
        colormap: str = "viridis",
        simplify_tolerance: int = 300,
    ):
        """Initialize the HRU Comparison Panel."""
        # Initialize Panel and HoloViews extensions
        pn.extension()
        hv.extension("bokeh")
        gv.extension("bokeh")

        # Store configuration
        self.shapefile_path = pl.Path(shapefile_path)
        self.variable_names = variable_names
        self.run_directories = {
            name: pl.Path(path) for name, path in run_directories.items()
        }
        self.run_names = list(self.run_directories.keys())

        # Visual parameters
        self.map_width = map_width
        self.map_height = map_height
        self.timeseries_width = timeseries_width
        self.timeseries_height = timeseries_height
        self.default_colormap = colormap
        self.simplify_tolerance = simplify_tolerance

        # Load shapefile
        print(f"Loading shapefile from {self.shapefile_path}...")
        self.gdf = gpd.read_file(self.shapefile_path)

        # Set up CRS
        if self.gdf.crs is None:
            print("Warning: No CRS found, assuming EPSG:4326")
            self.gdf.set_crs(epsg=4326, inplace=True)

        print(f"Original CRS: {self.gdf.crs}")
        print("Reprojecting to Web Mercator (EPSG:3857)...")
        self.gdf = self.gdf.to_crs(epsg=3857)

        # Simplify geometries to speed up rendering (tolerance in meters for EPSG:3857)
        if self.simplify_tolerance > 0:
            print(
                f"Simplifying geometries with tolerance={self.simplify_tolerance}m for faster rendering..."
            )
            num_hrus = len(self.gdf)
            print(f"  Domain has {num_hrus} HRUs")
            if num_hrus > 1000:
                print(
                    f"  WARNING: Large domain ({num_hrus} HRUs) may render slowly."
                )
                print(
                    f"  Consider increasing simplify_tolerance (currently {self.simplify_tolerance}m) to 500-1000m"
                )
            self.gdf["geometry"] = self.gdf.geometry.simplify(
                tolerance=self.simplify_tolerance, preserve_topology=True
            )
            print("Geometries simplified")

        # Find HRU ID column
        if hru_id_column:
            self.hru_id_column = hru_id_column
        else:
            self.hru_id_column = self._detect_hru_id_column()

        print(f"Using HRU ID column: {self.hru_id_column}")
        print(f"Available shapefile columns: {list(self.gdf.columns)}")

        # Data storage
        self.data_cache = {}  # {(var_name, run_name): xr.DataArray}
        self.spatial_dim = None
        self.hru_ids = None

        # Variable metadata cache
        self.var_metadata = {}

        # Widgets (will be initialized in create_app)
        self.selected_hru_widget = None
        self.variable_selector = None
        self.left_run_selector = None
        self.right_run_selector = None

    def _detect_hru_id_column(self) -> str:
        """Auto-detect HRU ID column in shapefile."""
        possible_cols = [
            "nhm_id",
            "hru_id",
            "model_idx",
            "nhru_id",
            "GRID_CODE",
            "nhru",
            "hru",
        ]
        for col in possible_cols:
            if col in self.gdf.columns:
                return col
        raise ValueError(
            f"Could not auto-detect HRU ID column. "
            f"Available columns: {list(self.gdf.columns)}"
        )

    def load_variable_data(self, var_name: str, run_name: str) -> xr.DataArray:
        """
        Load data for a specific variable and run.

        Parameters
        ----------
        var_name : str
            Variable name
        run_name : str
            Run name

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        cache_key = (var_name, run_name)

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        nc_path = self.run_directories[run_name] / f"{var_name}.nc"

        if not nc_path.exists():
            raise FileNotFoundError(
                f"File not found: {nc_path}\n"
                f"Variable: {var_name}, Run: {run_name}"
            )

        print(f"Loading {var_name} from {run_name}...")
        da = xr.load_dataarray(nc_path)
        self.data_cache[cache_key] = da

        # Set spatial dimension and HRU IDs from first loaded variable
        if self.spatial_dim is None:
            self.spatial_dim = [d for d in da.dims if d != "time"][0]
            self.hru_ids = sorted(
                [int(x) for x in da[self.spatial_dim].values]
            )
            print(f"Spatial dimension: {self.spatial_dim}")
            print(f"Number of HRUs in NetCDF: {len(self.hru_ids)}")
            print(
                f"NetCDF HRU ID range: {min(self.hru_ids)} to {max(self.hru_ids)}"
            )

            # Check shapefile HRU IDs
            shp_hru_ids = sorted(self.gdf[self.hru_id_column].unique())
            print(f"Number of HRUs in shapefile: {len(shp_hru_ids)}")
            print(
                f"Shapefile HRU ID range: {min(shp_hru_ids)} to {max(shp_hru_ids)}"
            )

            # Check for mismatches
            in_shp_not_nc = set(shp_hru_ids) - set(self.hru_ids)
            in_nc_not_shp = set(self.hru_ids) - set(shp_hru_ids)
            if in_shp_not_nc:
                print(
                    f"WARNING: {len(in_shp_not_nc)} HRUs in shapefile but not in NetCDF"
                )
                print(f"  Examples: {list(in_shp_not_nc)[:5]}")
            if in_nc_not_shp:
                print(
                    f"WARNING: {len(in_nc_not_shp)} HRUs in NetCDF but not in shapefile"
                )
                print(f"  Examples: {list(in_nc_not_shp)[:5]}")

        return da

    def compute_time_mean(self, var_name: str, run_name: str) -> np.ndarray:
        """
        Compute time mean for a variable and run.

        Parameters
        ----------
        var_name : str
            Variable name
        run_name : str
            Run name

        Returns
        -------
        np.ndarray
            Time mean values for each HRU
        """
        da = self.load_variable_data(var_name, run_name)
        return da.mean(dim="time").values

    def compute_difference(
        self, var_name: str, left_run: str, right_run: str
    ) -> np.ndarray:
        """
        Compute difference between two runs (left - right).

        Parameters
        ----------
        var_name : str
            Variable name
        left_run : str
            Left run name
        right_run : str
            Right run name

        Returns
        -------
        np.ndarray
            Difference values for each HRU
        """
        left_mean = self.compute_time_mean(var_name, left_run)
        right_mean = self.compute_time_mean(var_name, right_run)
        return left_mean - right_mean

    def get_variable_metadata(self, var_name: str) -> dict:
        """
        Get metadata for a variable.

        Parameters
        ----------
        var_name : str
            Variable name

        Returns
        -------
        dict
            Dictionary with 'desc' and 'units' keys, or empty strings if not found
        """
        if var_name not in self.var_metadata:
            try:
                # Import metadata dynamically to avoid circular imports
                import pywatershed.base.meta as pws_meta

                meta_dict = pws_meta.find_variables(var_name)
                if var_name in meta_dict:
                    self.var_metadata[var_name] = {
                        "desc": meta_dict[var_name].get("desc", ""),
                        "units": meta_dict[var_name].get("units", ""),
                    }
                else:
                    self.var_metadata[var_name] = {"desc": "", "units": ""}
            except Exception:
                self.var_metadata[var_name] = {"desc": "", "units": ""}

        return self.var_metadata[var_name]

    def create_map_plot(
        self,
        var_name: str,
        left_run: str,
        right_run: Optional[str],
        cmap_name: str = "viridis",
        diff_tolerance: float = 0.0,
    ):
        """
        Create map plot showing either single run mean or difference.

        Parameters
        ----------
        var_name : str
            Variable name to plot
        left_run : str
            Left run name
        right_run : str or None
            Right run name (if None, plots only left run)

        Returns
        -------
        gv.Overlay
            GeoViews overlay with tiles and polygons
        """
        # Get variable metadata
        var_meta = self.get_variable_metadata(var_name)
        desc_str = f": {var_meta['desc']}" if var_meta["desc"] else ""
        units_str = f" ({var_meta['units']})" if var_meta["units"] else ""

        if right_run is None or right_run == "None":
            values = self.compute_time_mean(var_name, left_run)
            title = f"Time mean of {left_run}: {var_name}{desc_str}{units_str}"
            cmap = cmap_name
        else:
            values = self.compute_difference(var_name, left_run, right_run)
            title = f"Time mean difference of {left_run} - {right_run}: {var_name}{desc_str}{units_str}"
            cmap = cmap_name

        # Create a copy of gdf with values
        gdf_plot = self.gdf.copy()
        gdf_plot["value"] = gdf_plot[self.hru_id_column].map(
            lambda hru_id: values[self.hru_ids.index(hru_id)]
            if hru_id in self.hru_ids
            else np.nan
        )
        gdf_plot["hru_id_display"] = gdf_plot[self.hru_id_column]

        # Optionally filter out HRUs with small differences
        if diff_tolerance > 0 and (
            right_run is not None and right_run != "None"
        ):
            # Mask out values below the tolerance
            gdf_plot.loc[
                np.abs(gdf_plot["value"]) < diff_tolerance, "value"
            ] = np.nan

        # Create base tiles
        tiles = getattr(gv.tile_sources, self.tile_selector.value)()

        # Separate NaN and non-NaN polygons for different styling
        gdf_with_data = gdf_plot[~gdf_plot["value"].isna()].copy()
        gdf_nan = gdf_plot[gdf_plot["value"].isna()].copy()

        # Create polygons with data
        gv_polys_data = gv.Polygons(
            gdf_with_data,
            vdims=["value", "hru_id_display"],
            crs=ccrs.GOOGLE_MERCATOR,
        ).opts(
            color="value",
            cmap=cmap,
            colorbar=True,
            line_color="black",
            line_width=0.5,
            fill_alpha=0.7,
            tools=["tap", "hover"],
            active_tools=["tap"],
            hover_tooltips=[
                ("HRU ID", "@hru_id_display"),
                ("Value", "@value{0.000}"),
            ],
            selection_fill_color="yellow",
            selection_fill_alpha=0.7,
            selection_line_color="blue",
            selection_line_width=3,
            nonselection_alpha=0.5,
            nonselection_fill_alpha=0.3,
        )

        # Create transparent polygons for NaN values
        if len(gdf_nan) > 0:
            gv_polys_nan = gv.Polygons(
                gdf_nan,
                vdims=["hru_id_display"],
                crs=ccrs.GOOGLE_MERCATOR,
            ).opts(
                fill_alpha=0,  # Fully transparent
                line_color="lightgray",
                line_width=0.3,
                tools=["tap", "hover"],
                active_tools=["tap"],
                hover_tooltips=[
                    ("HRU ID", "@hru_id_display"),
                    ("Value", "No Data"),
                ],
                selection_fill_alpha=0.2,
                selection_line_color="blue",
                selection_line_width=3,
            )
            gv_polys = gv_polys_data * gv_polys_nan
        else:
            gv_polys = gv_polys_data

        # Combine and set options
        map_plot = (tiles * gv_polys).opts(
            width=self.map_width,
            height=self.map_height,
            title=title,
        )

        # Return the geodataframe with data for selection purposes
        return map_plot, gv_polys_data, gdf_with_data

    def create_timeseries_plot(
        self,
        var_name: str,
        hru_id: int,
    ):
        """
        Create timeseries plot for all runs at a selected HRU.

        Parameters
        ----------
        var_name : str
            Variable name
        hru_id : int
            HRU ID to plot

        Returns
        -------
        holoviews plot
            Timeseries plot
        """
        if hru_id not in self.hru_ids:
            return pn.pane.Markdown(f"**HRU {hru_id} not found in data**")

        # Collect timeseries for all runs
        data_dict = {}
        for run_name in self.run_names:
            da = self.load_variable_data(var_name, run_name)
            ts = da.sel({self.spatial_dim: hru_id})
            data_dict[run_name] = ts.values

        # Create DataFrame
        df = pd.DataFrame(
            data_dict,
            index=ts.coords["time"].values,
        )

        # Get variable metadata
        var_meta = self.get_variable_metadata(var_name)
        desc_str = f": {var_meta['desc']}" if var_meta["desc"] else ""
        ylabel = var_meta["units"] if var_meta["units"] else var_name

        # Create plot
        plot = df.hvplot.line(
            title=f"HRU {hru_id}: {var_name}{desc_str}",
            width=self.timeseries_width,
            height=self.timeseries_height,
            ylabel=ylabel,
            legend="top_right",
        )

        return plot

    def create_app(self):
        """
        Create and return the interactive Panel application.

        Returns
        -------
        pn.Column
            Panel application layout
        """
        # Initialize widgets
        self.variable_selector = pn.widgets.Select(
            name="Variable",
            options=self.variable_names,
            value=self.variable_names[0],
        )

        self.left_run_selector = pn.widgets.Select(
            name="Left Run (or single run if Right=None)",
            options=self.run_names,
            value=self.run_names[0],
        )

        right_options = ["None"] + self.run_names
        self.right_run_selector = pn.widgets.Select(
            name="Right Run (None = show only Left)",
            options=right_options,
            value="None" if len(self.run_names) < 2 else self.run_names[1],
        )

        # Colorblind-friendly colormap options
        colormap_options = {
            "Viridis (sequential)": "viridis",
            "Plasma (sequential)": "plasma",
            "Cividis (sequential)": "cividis",
            "Turbo (sequential)": "turbo",
            "Coolwarm (diverging)": "coolwarm",
            "RdYlBu (diverging)": "RdYlBu_r",
            "PiYG (diverging)": "PiYG",
            "BrBG (diverging)": "BrBG",
        }
        self.colormap_selector = pn.widgets.Select(
            name="Colormap",
            options=colormap_options,
            value=self.default_colormap,
        )

        # Numeric input for difference tolerance filter
        self.diff_tolerance = pn.widgets.FloatInput(
            name="Hide absolute diff below (0=off)",
            value=0.0,
            start=0.0,
            step=0.01,
            format="0.0000",
        )

        # Basemap tile options
        tile_options = {
            "OpenStreetMap": "OSM",
            "Satellite (ESRI)": "EsriImagery",
            "Terrain (Stamen)": "StamenTerrain",
            "Toner (Stamen)": "StamenToner",
            "CartoDB Positron": "CartoLight",
            "CartoDB Dark": "CartoDark",
        }
        self.tile_selector = pn.widgets.Select(
            name="Basemap",
            options=tile_options,
            value="EsriImagery",
        )

        # Load first variable to get HRU IDs
        _ = self.load_variable_data(self.variable_names[0], self.run_names[0])

        self.selected_hru_widget = pn.widgets.IntInput(
            name="Selected HRU ID",
            value=self.hru_ids[0],
            start=min(self.hru_ids),
            end=max(self.hru_ids),
        )

        # Store selection stream reference
        self._selection_stream = None
        self._gdf_for_selection = self.gdf  # Will be updated when map changes

        # Create Row containers that will hold single plot
        map_row = pn.Row(sizing_mode="stretch_width")
        timeseries_row = pn.Row(sizing_mode="stretch_width")

        # Function to update map
        def update_map(event=None):
            try:
                map_plot, gv_polys, self._gdf_for_selection = (
                    self.create_map_plot(
                        self.variable_selector.value,
                        self.left_run_selector.value,
                        self.right_run_selector.value,
                        self.colormap_selector.value,
                        self.diff_tolerance.value,
                    )
                )

                # Set up selection stream
                if self._selection_stream is None:
                    self._selection_stream = streams.Selection1D(
                        source=gv_polys
                    )

                    def update_from_selection(index):
                        if index and len(index) > 0:
                            idx = index[0]
                            if 0 <= idx < len(self._gdf_for_selection):
                                hru_id = int(
                                    self._gdf_for_selection.iloc[idx][
                                        self.hru_id_column
                                    ]
                                )
                                if hru_id in self.hru_ids:
                                    self.selected_hru_widget.value = hru_id

                    self._selection_stream.param.watch(
                        lambda event: update_from_selection(event.new), "index"
                    )
                else:
                    # Update the source for the existing selection stream
                    self._selection_stream.source = gv_polys

                # Update the row
                map_row.objects = [map_plot]

            except Exception as e:
                map_row.objects = [
                    pn.pane.Markdown(f"**Error creating map:** {str(e)}")
                ]

        # Function to update timeseries
        def update_timeseries(event=None):
            try:
                ts_plot = self.create_timeseries_plot(
                    self.variable_selector.value,
                    self.selected_hru_widget.value,
                )

                # Update the row
                timeseries_row.objects = [ts_plot]

            except Exception as e:
                timeseries_row.objects = [
                    pn.pane.Markdown(
                        f"**Error creating timeseries:** {str(e)}"
                    )
                ]

        # Create wrapper functions that show loading immediately
        def on_variable_change(event):
            map_row.objects = [pn.pane.Markdown("## ⏳ Loading map...")]
            timeseries_row.objects = [
                pn.pane.Markdown("## ⏳ Loading timeseries...")
            ]
            update_map(event)
            update_timeseries(event)

        def on_left_run_change(event):
            map_row.objects = [pn.pane.Markdown("## ⏳ Loading map...")]
            update_map(event)

        def on_right_run_change(event):
            map_row.objects = [pn.pane.Markdown("## ⏳ Loading map...")]
            update_map(event)

        def on_colormap_change(event):
            map_row.objects = [pn.pane.Markdown("## ⏳ Loading map...")]
            update_map(event)

        def on_diff_tolerance_change(event):
            map_row.objects = [pn.pane.Markdown("## ⏳ Loading map...")]
            update_map(event)

        def on_tile_change(event):
            map_row.objects = [pn.pane.Markdown("## ⏳ Loading map...")]
            update_map(event)

        def on_hru_change(event):
            timeseries_row.objects = [
                pn.pane.Markdown("## ⏳ Loading timeseries...")
            ]
            update_timeseries(event)

        # Watch widgets for changes
        self.variable_selector.param.watch(on_variable_change, "value")
        self.left_run_selector.param.watch(on_left_run_change, "value")
        self.right_run_selector.param.watch(on_right_run_change, "value")
        self.colormap_selector.param.watch(on_colormap_change, "value")
        self.diff_tolerance.param.watch(on_diff_tolerance_change, "value")
        self.tile_selector.param.watch(on_tile_change, "value")
        self.selected_hru_widget.param.watch(on_hru_change, "value")

        # Initialize plots
        update_map()
        update_timeseries()

        # Build layout with controls|map on top, timeseries full width below
        app = pn.Column(
            "# HRU Comparison Panel",
            f"**Comparing {len(self.run_names)} runs across {len(self.variable_names)} variables**",
            "**Click on any HRU polygon on the map to view its timeseries, or manually enter an HRU ID.**",
            pn.Row(
                pn.Column(
                    "### Plot Controls",
                    self.variable_selector,
                    self.left_run_selector,
                    self.right_run_selector,
                    self.colormap_selector,
                    self.tile_selector,
                    self.diff_tolerance,
                    "### HRU Selection",
                    self.selected_hru_widget,
                    "Click on map to select HRU",
                    width=300,
                ),
                pn.Column(
                    "## Spatial Pattern (Time Mean)",
                    map_row,
                ),
            ),
            "## Timeseries at Selected HRU",
            timeseries_row,
        )

        return app

    def show(self):
        """Create and display the app in a notebook."""
        app = self.create_app()
        return app.show()

    def servable(self):
        """Create and return servable app for deployment."""
        app = self.create_app()
        return app.servable()


# Convenience function for quick usage
def compare_hru_runs(
    shapefile_path: Union[str, pl.Path],
    variable_names: List[str],
    run_directories: Dict[str, Union[str, pl.Path]],
    **kwargs,
):
    """
    Convenience function to quickly create and display an HRU comparison app.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to shapefile containing HRU polygons
    variable_names : list of str
        List of variable names to compare
    run_directories : dict
        Dictionary mapping run names to directory paths
    **kwargs
        Additional keyword arguments passed to HRUComparisonPanel

    Returns
    -------
    pn.Column
        Panel application

    Examples
    --------
    >>> app = compare_hru_runs(
    ...     shapefile_path="model_nhru.shp",
    ...     variable_names=["prcp", "tmean"],
    ...     run_directories={"Run1": "/path/to/run1", "Run2": "/path/to/run2"},
    ... )
    >>> app.show()
    """
    comparer = HRUComparisonPanel(
        shapefile_path=shapefile_path,
        variable_names=variable_names,
        run_directories=run_directories,
        **kwargs,
    )
    return comparer.create_app()
