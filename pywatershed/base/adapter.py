"""The adapter module.

This module contains the Adapter base class, its several concrete subclasses
and an adapter factory to dispatch you the right subclass when you ask for it.

"""

import pathlib as pl
from typing import Union

import numpy as np
import pyPRMS as pyprms

from ..base.control import Control
from ..base.timeseries import TimeseriesArray
from ..constants import fileish, nan
from ..utils.netcdf_utils import NetCdfRead


class Adapter:
    """Adapter base class for getting data from a variety of sources.

    Args:
        variable: string name of variable
    """

    def __init__(
        self,
        variable: str,
    ) -> None:
        if not hasattr(self, "name"):
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


class AdapterDatafile(Adapter):
    """Adapter subclass for a NetCDF file

    This requires that the NetCDF file have a time dimension named "time" or
    "doy" (day of year) to be properly handled as a timeseries for input, etc.

    Args:
        fname: filename of netcdf as string or Path
        variable: variable name string
        control: a Control object
        load_n_time_batches: number of times to read from file.

    """

    def __init__(
        self,
        fname: fileish,
        variable: str,
        control: Control,
        missing: tuple[str] = ("-99.9", "-901.0", "-999.0", "-9999.0"),
    ) -> None:
        super().__init__(variable)
        self.name = "AdapterNetcdf"

        self._fname = fname
        self._variable = variable
        self.control = control

        data = pyprms.DataFile(self._fname, missing=missing).data
        data = data.loc[:, data.columns.str.match(self._variable)]
        if len(data.columns) == 0:
            raise ValueError(
                f"Variable '{variable}' not found in file {fname}"
            )

        self.time = data.index.values
        self._data = data.to_numpy()
        del data
        self._nstations = self._data.shape[1]

        self._current_time = self.control.init_time.copy()
        # should use constants.fill_values for these
        self._current_value = np.full(self._nstations, nan, self._data.dtype)
        # _time_index is to be advanced, so subtract one
        self._time_index = (
            np.where(self.time == self.control.start_time)[0][0] - 1
        )

        return

    def advance(self):
        if self._current_time >= self.control.current_time:
            return

        self._time_index += 1
        self._current_time = self.time[self._time_index]
        assert self.control.current_time == self._current_time

        self._current_value[:] = self._data[self._time_index, :]
        return None

    @property
    def data(self) -> np.array:
        """Return the data for all time."""
        return self._data


class AdapterNetcdf(Adapter):
    """Adapter subclass for a PRMS 'data file'.

    This brings nstations of the selected variable for all time.

    Args:
        fname: path of the data file with extension '.data'
        variable: variable name string
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
        self._current_value[:] = np.array(
            self._nc_read.advance(self._variable, self.control.current_time)
        )
        return None

    @property
    def data(self) -> np.array:
        """Return the data for all time."""
        # TODO JLM: seems like we'd want to cache this data if we invoke once
        return self._nc_read.all_time(self._variable).data


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


adaptable = Union[str, pl.Path, np.ndarray, Adapter]


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
        # Paths and strings are considered paths to netcdf files
        if pl.Path(var).suffix == ".nc":
            return AdapterNetcdf(
                var,
                variable=variable_name,
                control=control,
                load_n_time_batches=load_n_time_batches,
            )
        elif pl.Path(var).suffix == ".data":
            return AdapterDatafile(
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
