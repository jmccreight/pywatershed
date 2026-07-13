"""The adapter module.

This module contains the Adapter base class, its several concrete subclasses
and an adapter factory to dispatch you the right subclass when you ask for it.

"""

import datetime
import pathlib as pl
import warnings
from typing import Union

import numpy as np

from ..base.control import Control
from ..base.timeseries import TimeseriesArray
from ..constants import fileish
from ..utils.netcdf_utils import NetCdfRead
from ..utils.prms_dyn_param import PrmsDynamicParameter


class Adapter:
    """Adapter base class for getting data from a variety of sources.

    Args:
        variable: string name of variable
    """

    def __init__(
        self,
        variable: str,
    ) -> None:
        self.name = "Adapter"
        self._variable = variable
        return None

    def advance(self) -> None:
        """Advance the adapter in time"""
        raise NotImplementedError("Must be overridden")

    @property
    def current(self):
        """Current time of the Adapter instance."""
        return self._current_value


class AdapterNetcdf(Adapter):
    """Adapter subclass for a NetCDF file

    This requires that the NetCDF file have a time dimension named "time" or
    "doy" (day of year) to be properly handled as a timeseries for input, etc.

    Args:
        fname: filename of netcdf as string or Path
        variable: variable name string
        dim_sizes: a tuple of dimension sizes
        type: a variable dtype
        control: a Control object
        load_n_time_batches: number of times to read from file.

    """

    def __init__(
        self,
        fname: fileish,
        variable: str,
        control: Control,
        load_n_time_batches: int = 1,
    ) -> None:
        super().__init__(variable)
        self.name = "AdapterNetcdf"

        self._fname = fname
        self.control = control
        self._start_time = self.control.start_time
        self._end_time = self.control.end_time

        self._nc_read = NetCdfRead(
            fname,
            start_time=self._start_time,
            end_time=self._end_time,
            load_n_time_batches=load_n_time_batches,
        )

        # would like to make this a check if dim_sizes and type are available
        nc_type = self._nc_read.dataset[variable].dtype
        nc_shape = list(self._nc_read.dataset[variable].shape)
        nc_dims = list(self._nc_read.dataset[variable].dimensions)
        for time_dim in ["time", "doy"]:
            if time_dim in nc_dims:
                _ = nc_shape.pop(nc_dims.index(time_dim))

        self.time = self._nc_read.times

        if "int" in str(nc_type):
            fill_value = -9999
        else:
            fill_value = np.nan

        self._current_value = np.full(nc_shape, fill_value, nc_type)

        return

    def advance(self):
        if self._nc_read._itime_step[self._variable] > self.control.itime_step:
            return
        self._current_value[:] = self._nc_read.advance(
            self._variable, self.control.current_time
        )
        return None

    @property
    def data(self) -> np.array:
        """Return the data for the current time."""
        # TODO JLM: seems like we'd want to cache this data if we invoke once
        return self._nc_read.all_time(self._variable).data

    def close(self):
        """Close the underlying NetCDF file."""
        if hasattr(self, "_nc_read"):
            self._nc_read.close()
        return


class AdapterOnedarray(Adapter):
    """Adapter subclass for an invariant 1-D numpy.array

    The data are constant and do not advance in time.

    Args:
        data: the data to be adapted
        variable: variable name string
    """

    def __init__(
        self,
        data: np.ndarray,
        variable: str,
        control: Control = None,
    ) -> None:
        super().__init__(variable)
        self.name = "AdapterOnedarray"
        self.control = control
        self._current_value = data
        return

    def advance(self, *args) -> None:
        """A dummy method for compliance."""
        return None


class AdapterDynamicParameter(Adapter):
    """Adapter subclass for PRMS dynamic parameter files.

    Dynamic parameter files contain parameter values that change over time,
    typically on a yearly basis. The adapter tracks the current date and
    updates the values when the date matches (or passes) the next available
    date in the file.

    Args:
        dyn_param: a PrmsDynamicParameter object or path to a dynamic
            param file
        variable: variable name string
        control: a Control object
        dtype: data type for loading file ('float' or 'int'), only used if
            dyn_param is a path

    """

    def __init__(
        self,
        dyn_param: Union[PrmsDynamicParameter, str, pl.Path],
        variable: str,
        control: Control,
        dtype: str = "float",
    ) -> None:
        super().__init__(variable)
        self.name = "AdapterDynamicParameter"
        self.control = control

        # Load from file if path provided
        if isinstance(dyn_param, (str, pl.Path)):
            self._dyn_param = PrmsDynamicParameter.load(dyn_param, dtype=dtype)
            self._fname = dyn_param
        else:
            self._dyn_param = dyn_param
            self._fname = None

        # Find starting index based on control.start_time
        self._date_index = self._find_date_index(control.start_time)
        self._current_value = self._dyn_param.data[self._date_index, :].copy()

        return

    def _find_date_index(self, target_date: np.datetime64) -> int:
        """Find the index of the date entry that applies for target_date.

        Returns the index of the most recent date that is <= target_date.
        """
        target_dt = target_date.astype("datetime64[D]").astype(datetime.date)
        if isinstance(target_dt, np.datetime64):
            # Handle numpy datetime64
            target_dt = target_dt.astype("M8[D]").astype("O")

        # Convert to datetime.date if needed
        if hasattr(target_date, "year"):
            target_dt = datetime.date(
                target_date.year, target_date.month, target_date.day
            )
        else:
            # numpy datetime64
            ts = (
                target_date - np.datetime64("1970-01-01", "D")
            ) / np.timedelta64(1, "D")
            dt = datetime.date.fromordinal(int(ts) + 719163)
            target_dt = dt

        # Find the most recent date <= target_date
        best_index = 0
        for i, date_row in enumerate(self._dyn_param.dates):
            file_date = datetime.date(
                int(date_row[0]), int(date_row[1]), int(date_row[2])
            )
            if file_date <= target_dt:
                best_index = i
            else:
                break

        return best_index

    def advance(self) -> None:
        """Advance the adapter, updating values if a new date is reached."""
        # Check if there's a next date entry
        if self._date_index + 1 >= len(self._dyn_param.dates):
            return

        # Get the next date in the file
        next_date_row = self._dyn_param.dates[self._date_index + 1]
        next_date = datetime.date(
            int(next_date_row[0]), int(next_date_row[1]), int(next_date_row[2])
        )

        # Get current simulation date
        current_dt = self.control.current_datetime
        current_date = datetime.date(
            current_dt.year, current_dt.month, current_dt.day
        )

        # If current date >= next date in file, advance to next values
        if current_date >= next_date:
            self._date_index += 1
            self._current_value[:] = self._dyn_param.data[self._date_index, :]

        return None


adaptable = Union[str, pl.Path, np.ndarray, Adapter, PrmsDynamicParameter]


def adapter_factory(
    var: adaptable,
    variable_name: str = None,
    control: Control = None,
    load_n_time_batches: int = 1,
) -> "Adapter":
    """A function to return the appropriate subclass of Adapter

    Args:
       var: the quantity to be adapted
       variable_name: what you call the above var
       control: a Control object
       variable_dim_sizes: for an AdapterNetcdf
       variable_type: for an AdapterNetcdf
       load_n_time_batches: for an AdapterNetcdf

    """
    if isinstance(var, Adapter):
        # Adapt an adapter.
        return var

    elif isinstance(var, (str, pl.Path)):
        path = pl.Path(var)
        # Paths and strings are considered paths to netcdf files
        if path.suffix == ".nc":
            return AdapterNetcdf(
                var,
                variable=variable_name,
                control=control,
                load_n_time_batches=load_n_time_batches,
            )
        # Dynamic parameter files (.param or .dyn)
        elif path.suffix in (".param", ".dyn"):
            # there is some danger here with the regular parameter file, but
            # that should trhow some errors
            warnings.warn(f"Note: Using dynamic parameter file {str(path)}")
            return AdapterDynamicParameter(
                var,
                variable=variable_name,
                control=control,
            )

    elif isinstance(var, PrmsDynamicParameter):
        # Adapt a PrmsDynamicParameter object directly
        return AdapterDynamicParameter(
            var,
            variable=variable_name,
            control=control,
        )

    elif isinstance(var, np.ndarray) and len(var.shape) == 1:
        # Adapt 1-D np.ndarrays
        return AdapterOnedarray(var, variable=variable_name, control=control)

    elif isinstance(var, TimeseriesArray):
        # Adapt TimeseriesArrays as is.
        return var

    elif var is None:
        # var is specified as None so return None
        return None

    else:
        raise TypeError("oops you screwed up")
