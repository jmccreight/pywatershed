"""Time series statistical functions for pywatershed analysis.

This module provides statistical functions designed to work with xarray
DataArrays, particularly for hydrological time series analysis. These
functions can be used with CustomOutput or independently for analysis.

All functions accept xarray DataArrays and return xarray DataArrays,
preserving coordinates and metadata where appropriate.

Examples
--------
>>> import xarray as xr
>>> from pywatershed.analysis.time_stats import (
...     mean,
...     seven_day_mean_water_year_min,
... )
>>>
>>> # Load some streamflow data
>>> flow_data = xr.open_dataset("streamflow.nc")["seg_outflow"]
>>>
>>> # Calculate mean over time
>>> mean_flow = mean(flow_data, dim="time")
>>>
>>> # Calculate 7-day low flows for each water year
>>> low_flows = seven_day_mean_water_year_min(flow_data)
"""

import xarray as xr

# Season strings for annual statistics
# Calendar year: January through December
cal_year_season_str = "JFMAMJJASOND"
# Water year: October through September
water_year_season_str = "ONDJFMAMJJAS"


def mean(
    da: xr.DataArray,
    dim: str | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Calculate mean along specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    dim : str, optional
        Dimension(s) over which to calculate mean
    skipna : bool, optional
        Whether to skip NaN values
    keep_attrs : bool, optional
        Whether to preserve attributes
    **kwargs
        Additional keyword arguments passed to xarray mean

    Returns
    -------
    xr.DataArray
        Mean values
    """
    return da.mean(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)


def std(
    da: xr.DataArray,
    dim: str | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Calculate standard deviation along specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    dim : str, optional
        Dimension(s) over which to calculate standard deviation
    skipna : bool, optional
        Whether to skip NaN values
    keep_attrs : bool, optional
        Whether to preserve attributes
    **kwargs
        Additional keyword arguments passed to xarray std

    Returns
    -------
    xr.DataArray
        Standard deviation values
    """
    return da.std(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)


def median(
    da: xr.DataArray,
    dim: str | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    **kwargs,
) -> xr.DataArray:
    """Calculate median along specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    dim : str, optional
        Dimension(s) over which to calculate median
    skipna : bool, optional
        Whether to skip NaN values
    keep_attrs : bool, optional
        Whether to preserve attributes
    **kwargs
        Additional keyword arguments passed to xarray median

    Returns
    -------
    xr.DataArray
        Median values
    """
    return da.median(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)


def seasonal_min_max(
    da: xr.DataArray,
    seasons: str | list = water_year_season_str,
    stat_name: str = "max",
) -> xr.DataArray:
    """Calculate seasonal extremes with dates of occurrence.

    Finds min/max over custom seasons and returns both values
    and dates when extremes occurred.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension
    seasons : str or list, default "ONDJFMAMJJAS"
        Season string(s) defining period(s). Default: water year (Oct-Sep).
        Use "JFMAMJJASOND" for calendar year (Jan-Dec).
        Can be a list for multiple seasons.
    stat_name : str, default "max"
        Statistic to calculate: "max" or "min"

    Returns
    -------
    xr.DataArray
        Seasonal extreme values with time coordinate replaced by dates
        when extremes occurred

    Examples
    --------
    >>> # Water year maximum
    >>> seasonal_max = seasonal_min_max(flow_data, stat_name="max")
    >>> # Calendar year minimum
    >>> seasonal_min = seasonal_min_max(
    ...     flow_data, seasons="JFMAMJJASOND", stat_name="min"
    ... )
    """
    if not isinstance(seasons, list):
        seasons = [seasons]
    if stat_name == "max":
        stat = da.resample(time=xr.groupers.SeasonResampler(seasons)).max(
            dim="time", skipna=True
        )
        stat_dates = da.resample(
            time=xr.groupers.SeasonResampler(seasons)
        ).map(lambda x: x.idxmax(dim="time", skipna=True))
    elif stat_name == "min":
        stat = da.resample(time=xr.groupers.SeasonResampler(seasons)).min(
            dim="time", skipna=True
        )
        stat_dates = da.resample(
            time=xr.groupers.SeasonResampler(seasons)
        ).map(lambda x: x.idxmin(dim="time", skipna=True))
    else:
        raise ValueError(f"Stat '{stat_name}' not available.")
    # <
    stat["time"] = stat_dates
    if "units" in stat["time"].attrs:
        del stat["time"].attrs["units"]
    return stat


def rolling_mean_seasonal_min_max(
    da: xr.DataArray,
    time_window: int = 7,
    min_periods: int | None = None,
    center: bool = True,
    seasons: str | list = water_year_season_str,
    stat_name: str = "max",
) -> xr.DataArray:
    """Calculate seasonal extremes of rolling mean.

    Computes rolling mean, then finds seasonal min/max. Used for hydrological
    statistics like 7-day low/high flows.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension
    time_window : int, default 7
        Size of the rolling window in time steps (typically days)
    min_periods : int, optional
        Minimum number of observations required to have a value
    center : bool, default True
        Whether to center the rolling window
    seasons : str or list, default "ONDJFMAMJJAS"
        Season string(s) defining period(s). Default: water year.
        Can be a list for multiple seasons.
    stat_name : str, default "max"
        Statistic to calculate: "max" or "min"

    Returns
    -------
    xr.DataArray
        Seasonal extreme values of the rolling mean with dates

    See Also
    --------
    seven_day_mean_calendar_year_max : Calendar year 7-day high flows
    seven_day_mean_water_year_min : Water year 7-day low flows
    """
    if not isinstance(seasons, list):
        seasons = [seasons]
    roll_mean = da.rolling(
        time=time_window, min_periods=min_periods, center=center
    ).mean()
    stat = seasonal_min_max(roll_mean, seasons=seasons, stat_name=stat_name)
    return stat


def seven_day_mean_calendar_year_max(
    da: xr.DataArray,
    time_window: int = 7,
    min_periods: int | None = None,
    center: bool = True,
) -> xr.DataArray:
    """Calendar year 7-day high flow statistic.

    Computes highest 7-day average for each calendar year (Jan-Dec).
    Common for characterizing annual high flow conditions.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension (typically daily streamflow)
    time_window : int, default 7
        Size of the rolling window in days
    min_periods : int, optional
        Minimum number of observations required to have a value
    center : bool, default True
        Whether to center the rolling window

    Returns
    -------
    xr.DataArray
        Annual maximum 7-day mean values with dates when they occurred

    Examples
    --------
    >>> # Calculate 7-day high flows for each calendar year
    >>> high_flows = seven_day_mean_calendar_year_max(streamflow_data)
    """
    return rolling_mean_seasonal_min_max(
        da,
        time_window=time_window,
        min_periods=min_periods,
        center=center,
        seasons=cal_year_season_str,
        stat_name="max",
    )


def seven_day_mean_water_year_max(
    da: xr.DataArray,
    time_window: int = 7,
    min_periods: int | None = None,
    center: bool = True,
) -> xr.DataArray:
    """Water year 7-day high flow statistic.

    Computes highest 7-day average for each water year (Oct-Sep).
    Common in hydrological analysis and ecological flow assessments.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension (typically daily streamflow)
    time_window : int, default 7
        Size of the rolling window in days
    min_periods : int, optional
        Minimum number of observations required to have a value
    center : bool, default True
        Whether to center the rolling window

    Returns
    -------
    xr.DataArray
        Water year maximum 7-day mean values with dates when they occurred

    Notes
    -----
    Water year runs from October 1 through September 30, which better
    captures the hydrologic cycle in snow-dominated watersheds.

    Examples
    --------
    >>> # Calculate 7-day high flows for each water year
    >>> high_flows = seven_day_mean_water_year_max(streamflow_data)
    """
    return rolling_mean_seasonal_min_max(
        da,
        time_window=time_window,
        min_periods=min_periods,
        center=center,
        seasons=water_year_season_str,
        stat_name="max",
    )


def seven_day_mean_calendar_year_min(
    da: xr.DataArray,
    time_window: int = 7,
    min_periods: int | None = None,
    center: bool = True,
) -> xr.DataArray:
    """Calendar year 7-day low flow statistic.

    Computes lowest 7-day average for each calendar year (Jan-Dec).
    Related to "7Q10" statistic used for water quality standards.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension (typically daily streamflow)
    time_window : int, default 7
        Size of the rolling window in days
    min_periods : int, optional
        Minimum number of observations required to have a value
    center : bool, default True
        Whether to center the rolling window

    Returns
    -------
    xr.DataArray
        Annual minimum 7-day mean values with dates when they occurred

    Notes
    -----
    The 7-day low flow is an important metric for water quality regulations
    and ecological flow requirements. The "7Q10" statistic (10-year, 7-day
    low flow) is commonly used as a design flow for wastewater discharge
    permits.

    Examples
    --------
    >>> # Calculate 7-day low flows for each calendar year
    >>> low_flows = seven_day_mean_calendar_year_min(streamflow_data)
    """
    return rolling_mean_seasonal_min_max(
        da,
        time_window=time_window,
        min_periods=min_periods,
        center=center,
        seasons=cal_year_season_str,
        stat_name="min",
    )


def seven_day_mean_water_year_min(
    da: xr.DataArray,
    time_window: int = 7,
    min_periods: int | None = None,
    center: bool = True,
) -> xr.DataArray:
    """Water year 7-day low flow statistic.

    Computes lowest 7-day average for each water year (Oct-Sep).
    Used for low flow analysis and drought assessment.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension (typically daily streamflow)
    time_window : int, default 7
        Size of the rolling window in days
    min_periods : int, optional
        Minimum number of observations required to have a value
    center : bool, default True
        Whether to center the rolling window

    Returns
    -------
    xr.DataArray
        Water year minimum 7-day mean values with dates when they occurred

    Notes
    -----
    Water year runs from October 1 through September 30. Using water year
    for low flow analysis is common in regions where summer low flows are
    the critical period for aquatic ecosystems.

    Examples
    --------
    >>> # Calculate 7-day low flows for each water year
    >>> low_flows = seven_day_mean_water_year_min(streamflow_data)
    """
    return rolling_mean_seasonal_min_max(
        da,
        time_window=time_window,
        min_periods=min_periods,
        center=center,
        seasons=water_year_season_str,
        stat_name="min",
    )


def median_by_month(da: xr.DataArray) -> xr.DataArray:
    """Calculate median grouped by month.

    Groups data by calendar month and calculates the median for each month
    across all years. Useful for understanding seasonal patterns.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension

    Returns
    -------
    xr.DataArray
        Median values for each calendar month (1-12)

    Examples
    --------
    >>> # Get median flow for each month across all years
    >>> monthly_median = median_by_month(streamflow_data)
    """
    return da.groupby("time.month").median(dim="time")


def median_monthly(da: xr.DataArray) -> xr.DataArray:
    """Calculate median resampled to monthly frequency.

    Resamples data to month start frequency and calculates median for each
    month. Returns a time series with one value per month.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension

    Returns
    -------
    xr.DataArray
        Monthly median values

    Examples
    --------
    >>> # Get median flow for each month in the time series
    >>> monthly_series = median_monthly(streamflow_data)
    """
    return da.resample(time="1MS").median(dim="time")


def mean_monthly(da: xr.DataArray) -> xr.DataArray:
    """Calculate mean resampled to monthly frequency.

    Resamples data to month start frequency and calculates mean for each
    month. Returns a time series with one value per month.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension

    Returns
    -------
    xr.DataArray
        Monthly mean values

    Examples
    --------
    >>> # Get mean flow for each month in the time series
    >>> monthly_means = mean_monthly(streamflow_data)
    """
    return da.resample(time="1MS").mean(dim="time")


def max_5day(da: xr.DataArray) -> xr.DataArray:
    """Calculate maximum resampled to 5-day periods.

    Resamples data to 5-day periods and calculates the maximum for each
    period. Useful for identifying short-term peaks.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension

    Returns
    -------
    xr.DataArray
        Maximum values for each 5-day period

    Examples
    --------
    >>> # Get peak flow for each 5-day period
    >>> five_day_peaks = max_5day(streamflow_data)
    """
    return da.resample(time="5D").max(dim="time")


def max_yearly(da: xr.DataArray) -> xr.DataArray:
    """Calculate maximum resampled to yearly frequency.

    Resamples data to year start frequency and calculates the maximum for
    each year. Returns annual peak values.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension

    Returns
    -------
    xr.DataArray
        Annual maximum values

    Examples
    --------
    >>> # Get annual peak flow
    >>> annual_peaks = max_yearly(streamflow_data)
    """
    return da.resample(time="1YS").max(dim="time")


__all__ = [
    "mean",
    "std",
    "median",
    "seasonal_min_max",
    "rolling_mean_seasonal_min_max",
    "seven_day_mean_calendar_year_max",
    "seven_day_mean_water_year_max",
    "seven_day_mean_calendar_year_min",
    "seven_day_mean_water_year_min",
    "median_by_month",
    "median_monthly",
    "mean_monthly",
    "max_5day",
    "max_yearly",
    "cal_year_season_str",
    "water_year_season_str",
]
