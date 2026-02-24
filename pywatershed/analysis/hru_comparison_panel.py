#!/usr/bin/env python
# coding: utf-8

"""
HRU Comparer - Interactive comparison of model runs by HRU

This module provides a flexible class for comparing multiple model runs
with interactive visualization of spatial patterns and timeseries.
"""

import pathlib as pl
from typing import Dict, List, Optional, Union

# import geopandas as gpd
# import geoviews as gv
# import holoviews as hv
import numpy as np
import pandas as pd

# import panel as pn
import xarray as xr

# from bokeh.models import DatetimeTickFormatter
# from cartopy import crs as ccrs
# from holoviews import streams
from ..utils.optional_import import import_optional_dependency

pn = import_optional_dependency("panel")
gv = import_optional_dependency("geoviews")
hv = import_optional_dependency("holoviews")
gpd = import_optional_dependency("geopandas")
streams = import_optional_dependency("holoviews.streams")
ccrs = import_optional_dependency("cartopy.crs")
bokeh_models = import_optional_dependency("bokeh.models")
if bokeh_models is not None:
    DatetimeTickFormatter = bokeh_models.DatetimeTickFormatter
else:
    DatetimeTickFormatter = None
_ = import_optional_dependency("hvplot.pandas")


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
    input_directories : dict, optional
        Dictionary mapping run names to input directory paths. If a variable is not
        found in run_directories, will check input_directories.
        Example: {"Run1": "/path/to/run1/input", "Run2": "/path/to/run2/input"}
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
        Tolerance in meters for geometry simplification (default: 300).
        Increase this (e.g., 500 or 1000) for faster rendering with large domains.
    time_aggregations : dict, optional
        Dictionary mapping aggregation names to functions that take an xarray
        DataArray and return aggregated values. If None, uses built-in defaults:
        Mean, Sum, Max, Min, Std, Range, Trend.
    run_colors : dict, optional
        Dictionary mapping run names to colors for timeseries plots.
        If None, uses built-in colorblind-friendly palette.
        Example: {"Run1": "#0173B2", "Run2": "#DE8F05"}
        Custom examples:
            {
                "Median": lambda da: da.median(dim="time").values,
                "95th percentile": lambda da: da.quantile(0.95, dim="time").values,
                "Jan-Mar Mean": lambda da: da.sel(
                    time=da.time.dt.month.isin([1,2,3])
                ).mean(dim="time").values,
            }

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
        input_directories: Optional[Dict[str, Union[str, pl.Path]]] = None,
        hru_id_column: Optional[str] = "nhru_v1_1",
        map_width: int = 1200,
        map_height: int = 650,
        timeseries_width: int = 1500,
        timeseries_height: int = 250,
        colormap: str = "viridis",
        simplify_tolerance: int = 300,
        time_aggregations: Optional[Dict] = None,
        run_colors: Optional[Dict[str, str]] = None,
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
        self.input_directories = (
            {name: pl.Path(path) for name, path in input_directories.items()}
            if input_directories
            else {}
        )
        self.run_names = list(self.run_directories.keys())

        # Assign consistent colors to each run
        if run_colors:
            # Validate that all runs have colors specified
            missing_runs = set(self.run_names) - set(run_colors.keys())
            if missing_runs:
                raise ValueError(
                    f"run_colors must include all runs. Missing colors for: {missing_runs}"
                )
            self.run_colors = run_colors
        else:
            # Use default colorblind-friendly palette
            color_palette = [
                "#0173B2",  # blue
                "#DE8F05",  # orange
                "#029E73",  # green
                "#CC78BC",  # purple
                "#CA9161",  # brown
                "#949494",  # gray
                "#ECE133",  # yellow
                "#56B4E9",  # light blue
            ]
            self.run_colors = {
                run_name: color_palette[i % len(color_palette)]
                for i, run_name in enumerate(self.run_names)
            }

        # Set up time aggregations
        if time_aggregations is None:
            # Default aggregations
            self.time_aggregations = {
                "Mean": lambda da: da.mean(dim="time").values,
                "Sum": lambda da: da.sum(dim="time").values,
                "Max": lambda da: da.max(dim="time").values,
                "Min": lambda da: da.min(dim="time").values,
                "Std": lambda da: da.std(dim="time").values,
                "Range": lambda da: (
                    (da.max(dim="time") - da.min(dim="time")).values
                ),
                "Trend": lambda da: self._compute_trend(da),
            }
        else:
            # User-provided custom aggregations
            self.time_aggregations = time_aggregations

        self.aggregation_names = list(self.time_aggregations.keys())

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

        # Check which runs have which variables and diagnose dimension issues
        print("Checking variable availability across runs...")
        self.var_availability = {}  # {var_name: [run_names that have it]}
        self.var_locations = {}  # {(var_name, run_name): 'run'|'input'}

        for var_name in self.variable_names:
            available_runs = []
            for run_name in self.run_names:
                # Check run directory first
                nc_path = self.run_directories[run_name] / f"{var_name}.nc"
                if nc_path.exists():
                    available_runs.append(run_name)
                    self.var_locations[(var_name, run_name)] = "run"
                    self._diagnose_file(nc_path, var_name, run_name, "run")
                # Check input directory if not found in run
                elif run_name in self.input_directories:
                    input_path = (
                        self.input_directories[run_name] / f"{var_name}.nc"
                    )
                    if input_path.exists():
                        available_runs.append(run_name)
                        self.var_locations[(var_name, run_name)] = "input"
                        self._diagnose_file(
                            input_path, var_name, run_name, "input"
                        )

            self.var_availability[var_name] = available_runs
            if available_runs:
                print(
                    f"  {var_name}: available in {', '.join(available_runs)}"
                )
            else:
                print(f"  {var_name}: NOT FOUND in any run")

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

    def load_variable_data(
        self, var_name: str, run_name: str
    ) -> Optional[xr.DataArray]:
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
        xr.DataArray or None
            Loaded data array, or None if file doesn't exist
        """
        cache_key = (var_name, run_name)

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Determine which directory to load from
        if (var_name, run_name) in self.var_locations:
            if self.var_locations[(var_name, run_name)] == "input":
                nc_path = self.input_directories[run_name] / f"{var_name}.nc"
            else:
                nc_path = self.run_directories[run_name] / f"{var_name}.nc"
        else:
            nc_path = self.run_directories[run_name] / f"{var_name}.nc"

        if not nc_path.exists():
            # Return None instead of raising error
            self.data_cache[cache_key] = None
            return None

        print(f"Loading {var_name} from {run_name}...")
        da = xr.load_dataarray(nc_path)

        # Check if this file uses index-based dimension (nhru) with nhm_id coordinate
        spatial_dim = [d for d in da.dims if d != "time"][0]
        if spatial_dim == "nhru" and "nhm_id" in da.coords:
            # This file uses indices - create a mapping
            nhm_id_to_index = {
                int(nhm_id): idx
                for idx, nhm_id in enumerate(da.coords["nhm_id"].values)
            }
            # Store the mapping with the cache key
            self.data_cache[f"{cache_key}_mapping"] = nhm_id_to_index
            print(
                "  File uses index-based dimension with nhm_id coordinate mapping"
            )

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

    def _diagnose_file(
        self, nc_path: pl.Path, var_name: str, run_name: str, location: str
    ):
        """Diagnose dimension and coordinate issues in a NetCDF file."""
        try:
            da = xr.load_dataarray(nc_path)
            dims = list(da.dims)
            coords = list(da.coords.keys())

            print(f"    [{location}] {var_name} in {run_name}:")
            print(f"      Dimensions: {dims}")
            print(f"      Coordinates: {coords}")

            # Check for spatial dimension
            spatial_dims = [d for d in dims if d != "time"]
            if spatial_dims:
                spatial_dim = spatial_dims[0]
                print(
                    f"      Spatial dimension: '{spatial_dim}' with {da.sizes[spatial_dim]} elements"
                )

                # Check if dimension name is nhm_id vs nhru
                if spatial_dim == "nhm_id":
                    print(
                        "      ⚠️  WARNING: Dimension is 'nhm_id', should be 'nhru'"
                    )
                    print(
                        f"      Coordinate values: {da[spatial_dim].values[:5]}... (first 5)"
                    )
                elif spatial_dim == "nhru":
                    print("      ✓ Dimension is 'nhru' (index-based)")
                    print(f"      Range: 0 to {da.sizes[spatial_dim] - 1}")
                else:
                    print(
                        f"      ⚠️  WARNING: Unexpected spatial dimension name: '{spatial_dim}'"
                    )

                # Check if nhm_id exists as a coordinate
                if "nhm_id" in coords and spatial_dim == "nhru":
                    print("      ✓ Has 'nhm_id' coordinate for mapping")
                    print(
                        f"      nhm_id values: {da.coords['nhm_id'].values[:5]}... (first 5)"
                    )

        except Exception as e:
            print(f"    ⚠️  Error diagnosing {nc_path}: {e}")

    def _compute_trend(self, da: xr.DataArray) -> np.ndarray:
        """Compute linear trend coefficient for each spatial location."""
        # Get time as numeric values (days since first timestep)
        time_numeric = (
            (da.coords["time"] - da.coords["time"][0]).values.astype(float)
            / 1e9
            / 86400
        )  # convert to days

        # Compute trend for each spatial location
        spatial_dim = [d for d in da.dims if d != "time"][0]
        trends = np.zeros(da.sizes[spatial_dim])

        for i in range(da.sizes[spatial_dim]):
            values = da.isel({spatial_dim: i}).values
            if not np.all(np.isnan(values)):
                # Simple linear regression: slope = cov(x,y) / var(x)
                trends[i] = np.cov(time_numeric, values)[0, 1] / np.var(
                    time_numeric
                )

        return trends

    def compute_time_aggregation(
        self, var_name: str, run_name: str, aggregation: str
    ) -> Optional[np.ndarray]:
        """
        Compute time aggregation for a variable and run.

        Parameters
        ----------
        var_name : str
            Variable name
        run_name : str
            Run name
        aggregation : str
            Aggregation method name

        Returns
        -------
        np.ndarray or None
            Aggregated values for each HRU, or None if data not available
        """
        da = self.load_variable_data(var_name, run_name)
        if da is None:
            return None

        agg_func = self.time_aggregations[aggregation]
        return agg_func(da)

    def compute_difference(
        self, var_name: str, left_run: str, right_run: str
    ) -> Optional[np.ndarray]:
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
        np.ndarray or None
            Difference values for each HRU, or None if either run doesn't have data
        """
        # Use current aggregation method
        agg_method = self._current_aggregation
        left_agg = self.compute_time_aggregation(
            var_name, left_run, agg_method
        )
        right_agg = self.compute_time_aggregation(
            var_name, right_run, agg_method
        )
        if left_agg is None or right_agg is None:
            return None
        return left_agg - right_agg

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
        aggregation: str = "Mean",
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
        # Store current aggregation for compute_difference to use
        self._current_aggregation = aggregation

        # Get variable metadata
        var_meta = self.get_variable_metadata(var_name)
        desc_str = f": {var_meta['desc']}" if var_meta["desc"] else ""
        units_str = f" ({var_meta['units']})" if var_meta["units"] else ""

        # Determine which runs to use based on availability
        available_runs = self.var_availability.get(var_name, [])

        if not available_runs:
            # No runs have this variable
            return (
                self._create_empty_map(
                    f"Variable '{var_name}' not found in any run"
                ),
                None,
                self.gdf,
            )

        # Determine actual left and right runs to use
        actual_left = left_run if left_run in available_runs else None
        actual_right = (
            right_run
            if (
                right_run
                and right_run != "None"
                and right_run in available_runs
            )
            else None
        )

        # If neither selected run has the variable, use first available
        if actual_left is None:
            actual_left = actual_right if actual_right else available_runs[0]
            actual_right = None

        # Compute values (single run or difference)
        if actual_right is None or actual_right == "None":
            values = self.compute_time_aggregation(
                var_name, actual_left, aggregation
            )
            title = f"{actual_left}: Temporal {aggregation}\n{var_name}{desc_str}{units_str}"
        else:
            values = self.compute_difference(
                var_name, actual_left, actual_right
            )
            title = f"{actual_left} - {actual_right}: Difference of temporal {aggregation}s\n{var_name}{desc_str}{units_str}"

        # Handle case where data could not be loaded
        if values is None:
            return (
                self._create_empty_map(f"Could not load data for {var_name}"),
                None,
                self.gdf,
            )

        cmap = cmap_name

        # Map values to geodataframe
        hru_id_to_idx = {
            hru_id: idx for idx, hru_id in enumerate(self.hru_ids)
        }
        gdf_plot = self.gdf.copy()
        gdf_plot["value"] = gdf_plot[self.hru_id_column].map(
            lambda hru_id: (
                values[hru_id_to_idx[hru_id]]
                if hru_id in hru_id_to_idx
                else np.nan
            )
        )
        gdf_plot["hru_id_display"] = gdf_plot[self.hru_id_column]

        # Optionally filter out HRUs with small differences
        if diff_tolerance > 0 and right_run and right_run != "None":
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

    def _create_empty_map(self, message: str):
        """Create an empty map with a message."""
        tiles = getattr(gv.tile_sources, self.tile_selector.value)()

        # Create empty polygons for outline only
        gv_polys = gv.Polygons(
            self.gdf,
            vdims=[self.hru_id_column],
            crs=ccrs.GOOGLE_MERCATOR,
        ).opts(
            fill_alpha=0,
            line_color="lightgray",
            line_width=0.5,
            tools=["hover"],
            hover_tooltips=[("HRU ID", f"@{self.hru_id_column}")],
        )

        map_plot = (tiles * gv_polys).opts(
            width=self.map_width,
            height=self.map_height,
            title=message,
        )

        return map_plot

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

        # Collect timeseries for runs that have this variable
        data_dict = {}
        time_coords = None
        for run_name in self.run_names:
            da = self.load_variable_data(var_name, run_name)
            if da is not None:
                # Find spatial dimension for this specific file
                spatial_dim = [d for d in da.dims if d != "time"][0]

                # Select HRU using index mapping if available
                try:
                    mapping_key = f"{(var_name, run_name)}_mapping"
                    if mapping_key in self.data_cache:
                        # Index-based file - use mapping
                        idx = self.data_cache[mapping_key].get(hru_id)
                        if idx is None:
                            continue
                        ts = da.isel({spatial_dim: idx})
                    else:
                        # ID-based file - use direct selection
                        ts = da.sel({spatial_dim: hru_id})

                    data_dict[run_name] = ts.values
                    if time_coords is None:
                        time_coords = ts.coords["time"].values
                except (KeyError, ValueError, IndexError):
                    continue

        if not data_dict:
            return pn.pane.Markdown(
                f"**Variable '{var_name}' not found in any run**"
            )

        # Create DataFrame
        df = pd.DataFrame(
            data_dict,
            index=time_coords,
        )

        # Get variable metadata
        var_meta = self.get_variable_metadata(var_name)
        desc_str = f": {var_meta['desc']}" if var_meta["desc"] else ""
        ylabel = var_meta["units"] if var_meta["units"] else var_name

        # Assign colors based on run names for consistency
        color_list = [self.run_colors[col] for col in df.columns]

        # Create plot with monthly x-axis ticks (month over year)
        plot = df.hvplot.line(
            title=f"HRU {hru_id}: {var_name}{desc_str}",
            width=self.timeseries_width,
            height=self.timeseries_height,
            ylabel=ylabel,
            legend="top_right",
            color=color_list,
        ).opts(
            xformatter=DatetimeTickFormatter(months="%b\n%Y"),
        )

        return plot

    def hru_plot_together(
        self,
        run_variables: Dict[str, List[str]],
        hru_id: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: Optional[str] = None,
        renamer: Optional[Dict[str, str]] = None,
        colors: Optional[Dict[str, str]] = None,
    ):
        """
        Plot multiple variables from different runs together at a single HRU.

        Variables with the same units are plotted on the same subplot, while
        variables with different units get separate subplots.

        Parameters
        ----------
        run_variables : dict
            Dictionary mapping run names to lists of variable names.
            Example: {"Run1": ["prcp", "hru_ppt"], "Run2": ["prcp"]}
        hru_id : int, optional
            HRU ID to plot. If None, uses current selected HRU from widget.
        width : int, optional
            Width of plot in pixels. If None, uses self.timeseries_width.
        height : int, optional
            Height of each subplot in pixels. If None, uses self.timeseries_height.
        title : str, optional
            Custom title to add after HRU ID. Replaces unit label in title.
            Example: "HRU 84966: Precipitation Analysis"
        renamer : dict, optional
            Dictionary to rename series labels in the legend.
            Keys are "{run}: {variable}", values are display names.
            Example: {"Run1: prcp": "Precipitation (Run1)"}
        colors : dict, optional
            Dictionary mapping series labels (after renaming) to colors.
            If None, uses run colors from self.run_colors.
            Example: {"Precipitation (Run1)": "#0173B2"}

        Returns
        -------
        hvplot object
            Plot or layout of plots

        Examples
        --------
        >>> plot = comparer.hru_plot_together(
        ...     {"Run1": ["prcp", "hru_ppt"], "Run2": ["prcp", "hru_rain"]},
        ...     hru_id=85806,
        ...     width=1600,
        ...     height=300,
        ...     renamer={
        ...         "Run1: prcp": "Precip",
        ...         "Run1: hru_ppt": "HRU Precip",
        ...     },
        ...     colors={"Precip": "#0173B2", "HRU Precip": "#DE8F05"},
        ... )
        """
        try:
            # Use current HRU if not specified
            if hru_id is None:
                if hasattr(self, "selected_hru_widget"):
                    hru_id = self.selected_hru_widget.value
                else:
                    raise ValueError(
                        "No HRU ID provided and no widget available"
                    )

            if hru_id not in self.hru_ids:
                return pn.pane.Markdown(f"**HRU {hru_id} not found in data**")

            # Set plot dimensions
            plot_width = width if width is not None else self.timeseries_width
            plot_height = (
                height if height is not None else self.timeseries_height
            )

            # Collect all timeseries with metadata
            series_data = []
            for run_name, var_list in run_variables.items():
                for var_name in var_list:
                    da = self.load_variable_data(var_name, run_name)
                    if da is None:
                        continue

                    # Get variable metadata
                    var_meta = self.get_variable_metadata(var_name)
                    units = var_meta.get("units", "")
                    desc = var_meta.get("desc", "")

                    # Extract timeseries for this HRU
                    spatial_dim = [d for d in da.dims if d != "time"][0]
                    try:
                        mapping_key = f"{(var_name, run_name)}_mapping"
                        if mapping_key in self.data_cache:
                            idx = self.data_cache[mapping_key].get(hru_id)
                            if idx is None:
                                continue
                            ts = da.isel({spatial_dim: idx})
                        else:
                            ts = da.sel({spatial_dim: hru_id})

                        series_data.append(
                            {
                                "run": run_name,
                                "variable": var_name,
                                "units": units,
                                "desc": desc,
                                "time": ts.coords["time"].values,
                                "values": ts.values,
                            }
                        )
                    except (KeyError, ValueError, IndexError):
                        continue

            if not series_data:
                return pn.pane.Markdown(
                    "**No data available for requested variables**"
                )

            # Group by units
            units_groups = {}
            for series in series_data:
                unit = series["units"]
                if unit not in units_groups:
                    units_groups[unit] = []
                units_groups[unit].append(series)

            # Create plots for each unit group
            plots = []
            for unit, group in units_groups.items():
                # Build dataframe for this unit group
                df_dict = {}
                time_index = None
                original_labels = []
                for series in group:
                    label = f"{series['run']}: {series['variable']}"
                    original_labels.append(label)
                    df_dict[label] = series["values"]
                    if time_index is None:
                        time_index = series["time"]

                df = pd.DataFrame(df_dict, index=time_index)

                # Apply renaming if provided
                if renamer:
                    df.rename(columns=renamer, inplace=True)

                # Build color mapping for all columns (after renaming)
                column_to_color = {}
                for i, col in enumerate(df.columns):
                    original_label = original_labels[i]
                    # Check colors dict for both renamed and original labels
                    if colors:
                        if col in colors:
                            # Use color for renamed label
                            column_to_color[col] = colors[col]
                        elif original_label in colors:
                            # Use color for original label
                            column_to_color[col] = colors[original_label]
                        else:
                            # Use run color as fallback
                            run_name = original_label.split(":")[0].strip()
                            column_to_color[col] = self.run_colors.get(
                                run_name, "#000000"
                            )
                    else:
                        # Use run colors
                        run_name = original_label.split(":")[0].strip()
                        column_to_color[col] = self.run_colors[run_name]

                # Reorder columns based on colors dict order if provided
                if colors:
                    ordered_cols = [
                        col for col in colors.keys() if col in df.columns
                    ]
                    remaining_cols = [
                        col for col in df.columns if col not in ordered_cols
                    ]
                    df = df[ordered_cols + remaining_cols]

                # Build color list and create plot
                color_list = [column_to_color[col] for col in df.columns]

                # Build title: HRU ID + custom title OR series name/unit
                single_series = len(df.columns) == 1
                if title:
                    plot_title = f"HRU {hru_id}: {title}"
                elif single_series:
                    plot_title = f"HRU {hru_id}: {df.columns[0]}"
                else:
                    unit_label = f" ({unit})" if unit else ""
                    plot_title = f"HRU {hru_id}{unit_label}"

                plot = df.hvplot.line(
                    title=plot_title,
                    ylabel=unit if unit else "Value",
                    legend=False if single_series else "top_right",
                    color=color_list,
                    width=plot_width,
                    height=plot_height,
                ).opts(xformatter=DatetimeTickFormatter(months="%b\n%Y"))

                plots.append(plot)

            # Return single plot or layout of plots
            if len(plots) == 1:
                return plots[0]
            else:
                # Return HoloViews Layout instead of Panel Column
                return hv.Layout(plots).cols(1)

        except Exception as e:
            return pn.pane.Markdown(
                f"**Error in hru_plot_together:** {str(e)}"
            )

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

        self.aggregation_selector = pn.widgets.Select(
            name="Time Aggregation",
            options=self.aggregation_names,
            value="Mean",
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

        # Function to update run selector options based on variable
        def update_run_selectors(event):
            var_name = self.variable_selector.value
            available_runs = self.var_availability.get(var_name, [])

            if not available_runs:
                # No runs have this variable
                self.left_run_selector.options = ["(none available)"]
                self.left_run_selector.value = "(none available)"
                self.right_run_selector.options = ["None"]
                self.right_run_selector.value = "None"
            else:
                # Update left selector
                if self.left_run_selector.value not in available_runs:
                    self.left_run_selector.value = available_runs[0]
                self.left_run_selector.options = available_runs

                # Update right selector
                right_opts = ["None"] + available_runs
                if self.right_run_selector.value not in right_opts:
                    self.right_run_selector.value = "None"
                self.right_run_selector.options = right_opts

        # Watch variable selector to update run options
        self.variable_selector.param.watch(update_run_selectors, "value")

        # Initialize run selectors for first variable
        update_run_selectors(None)

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

        # Load first available variable to get HRU IDs
        for var_name in self.variable_names:
            for run_name in self.run_names:
                da = self.load_variable_data(var_name, run_name)
                if da is not None:
                    break
            if self.hru_ids is not None:
                break

        if self.hru_ids is None:
            raise ValueError(
                "Could not load any variables from any runs. "
                "Please check that NetCDF files exist in the run directories."
            )

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
                result = self.create_map_plot(
                    self.variable_selector.value,
                    self.left_run_selector.value,
                    self.right_run_selector.value,
                    self.colormap_selector.value,
                    self.diff_tolerance.value,
                    self.aggregation_selector.value,
                )

                if result[1] is None:
                    # Empty map case (no data available)
                    map_plot = result[0]
                    map_row.objects = [map_plot]
                else:
                    map_plot, gv_polys, self._gdf_for_selection = result

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
                            lambda event: update_from_selection(event.new),
                            "index",
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

        def on_aggregation_change(event):
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
        self.aggregation_selector.param.watch(on_aggregation_change, "value")
        self.tile_selector.param.watch(on_tile_change, "value")
        self.selected_hru_widget.param.watch(on_hru_change, "value")

        # Initialize plots
        update_map()
        update_timeseries()

        # Build layout with controls|map on top, timeseries full width below
        app = pn.Column(
            "# HRU Comparison Panel",
            pn.pane.Markdown(
                f"**Comparing {len(self.run_names)} runs across {len(self.variable_names)} variables**  \n"
                f"Runs: {', '.join(self.run_names)}  \n"
                f"Variables: {', '.join(self.variable_names)}"
            ),
            "**Click on any HRU polygon on the map to view its timeseries, or manually enter an HRU ID.**",
            pn.Row(
                pn.Column(
                    "### Variable Selection",
                    self.variable_selector,
                    "### Spatial Plot Controls",
                    self.aggregation_selector,
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
                    "## Spatial pattern of stats for selected run(s)",
                    map_row,
                ),
            ),
            "## Timeseries for all (available) runs at selected HRU",
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
