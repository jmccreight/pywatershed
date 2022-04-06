import pathlib as pl
import numpy as np
from typing import Union

from pynhm.base.Time import Time
from ..utils.prms5util import load_nhru_output_csv

fileish = Union[str, pl.Path]


class NHMSnow(Time):
    def __init__(
        self,
        state_dict: dict,
        start_time: np.datetime64 = None,
        time_step: np.timedelta64 = None,  # could we infer timestep?
        end_time: np.datetime64 = None,
        datetime: np.ndarray = None,
        height_m: int = None,
        verbosity: int = 0,
    ):
        super().__init__(
            start_time=start_time,
            time_step=time_step,
            end_time=end_time,
            verbosity=verbosity,
        )
        self.name = "NHMSnow"
        self._coords += ["spatial_id"]

        self._potential_variables = [
            "pkwater_equiv",
            "snowcov_area",
            "snowmelt",
        ]

        self.spatial_id = None

        return

    @classmethod
    def load_prms_output(
        cls,
        pkwater_equiv: fileish = None,
        snowcov_area: fileish = None,
        snowmelt: fileish = None,
        **kwargs,
    ):
        """Instantiate an NHM snow modelfrom NHM/PRMS
        output csv files."""

        obj = cls({}, **kwargs)
        del obj.datetime

        obj.prms_output_files = {
            "pkwater_equiv": pkwater_equiv,
            "snowcov_area": snowcov_area,
            "snowmelt": snowmelt,
        }

        for var, var_file in obj.prms_output_files.items():
            if var_file is None:
                continue
            prms_output = load_nhru_output_csv(var_file)
            obj[var] = prms_output.to_numpy()
            if not hasattr(obj, "datetime"):
                obj["datetime"] = prms_output.index.to_numpy()
            else:
                assert (obj["datetime"] == prms_output.index.to_numpy()).all()

        return obj
