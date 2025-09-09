from typing import Literal

import numpy as np
import pandas as pd

from pywatershed.base.control import Control
from pywatershed.constants import cfs_to_cms, cms_to_cfs, nan1d, one, zero
from pywatershed.hydrology.starfit import (
    Starfit,
    StarfitFlowNode,
    StarfitFlowNodeMaker,
)
from pywatershed.parameters import Parameters


class StarfitSourceSinkFlowNode(StarfitFlowNode):
    """A StarfitFlowNode where sinks and sources interact with storage.

    James McCreight (UCAR/USGS), John Engott (USGS), and Noah Knowles (USGS)
    """

    def __init__(
        self,
        control: Control,
        grand_id: np.int64,
        initial_storage: np.float64,
        start_time: np.datetime64,
        end_time: np.datetime64,
        inflow_mean: np.float64,
        NORhi_min: np.float64,
        NORhi_max: np.float64,
        NORhi_alpha: np.float64,
        NORhi_beta: np.float64,
        NORhi_mu: np.float64,
        NORlo_min: np.float64,
        NORlo_max: np.float64,
        NORlo_alpha: np.float64,
        NORlo_beta: np.float64,
        NORlo_mu: np.float64,
        Release_min: np.float64,
        Release_max: np.float64,
        Release_alpha1: np.float64,
        Release_alpha2: np.float64,
        Release_beta1: np.float64,
        Release_beta2: np.float64,
        Release_p1: np.float64,
        Release_p2: np.float64,
        Release_c: np.float64,
        GRanD_CAP_MCM: np.float64,
        Obs_MEANFLOW_CUMECS: np.float64,
        source_sink_storage_min: np.float64,
        source_sink_data: pd.Series,
        missing_data_as_zero: bool = False,
        calc_method: Literal["numba", "numpy"] = None,
        io_in_cfs: bool = True,
        nhrs_substep: int = one,
        budget_type: Literal["defer", None, "warn", "error"] = None,
    ) -> None:
        """Initialize a StarfitSourceSinkFlowNode.

        Args:
            control: A Control object
            grand_id: the GRanD id,
            initial_storage: initial storage value, may be NaN to use the
                middle of the normal operating range.
            start_time: May be NaN, the optional time at which to start
                simulating with STARFIT.
            end_time: As for start_Time.
            inflow_mean: STARFIT parameter.
            NORhi_min: STARFIT parameter.
            NORhi_max: STARFIT parameter.
            NORhi_alpha: STARFIT parameter.
            NORhi_beta: STARFIT parameter.
            NORhi_mu: STARFIT parameter.
            NORlo_min: STARFIT parameter.
            NORlo_max: STARFIT parameter.
            NORlo_alpha: STARFIT parameter.
            NORlo_beta: STARFIT parameter.
            NORlo_mu: STARFIT parameter.
            Release_min: STARFIT parameter.
            Release_max: STARFIT parameter.
            Release_alpha1: STARFIT parameter.
            Release_alpha2: STARFIT parameter.
            Release_beta1: STARFIT parameter.
            Release_beta2: STARFIT parameter.
            Release_p1: STARFIT parameter.
            Release_p2: STARFIT parameter.
            Release_c: STARFIT parameter.
            GRanD_CAP_MCM: STARFIT parameter.
            Obs_MEANFLOW_CUMECS: STARFIT parameter.
            source_sink_storage_min: A floating point value for the minium
              flow.
            source_sink_data: A pandas Series object of sources/sinks at this
              location. See SourceSinkFlowNodeMaker for a description of the
              pd.DataFrame passed to supply this data. Units are cubic feet
              per second.
            missing_data_as_zero: Bool option to treat missing times in the
              timeseries as having zero source/sink.
            calc_method: One of "numba" or "numpy".
            io_in_cfs: Are the units in cubic feet per second? False gives
                units of cubic meters per second.
            nhrs_substep: Number of hours in the subtimestep.
            budget_type: One of "defer", "warn", or "error".
        """
        from datetime import datetime

        # needs to be above super.__init__ because of budget init in there.
        self._lake_storage_after_source_sink = nan1d()
        self._source_sink = nan1d()
        self._sink_source = nan1d()
        self._negative_sink_source = nan1d()
        self._source = nan1d()
        self._sink = nan1d()

        super().__init__(
            control=control,
            grand_id=grand_id,
            initial_storage=initial_storage,
            start_time=start_time,
            end_time=end_time,
            inflow_mean=inflow_mean,
            NORhi_min=NORhi_min,
            NORhi_max=NORhi_max,
            NORhi_alpha=NORhi_alpha,
            NORhi_beta=NORhi_beta,
            NORhi_mu=NORhi_mu,
            NORlo_min=NORlo_min,
            NORlo_max=NORlo_max,
            NORlo_alpha=NORlo_alpha,
            NORlo_beta=NORlo_beta,
            NORlo_mu=NORlo_mu,
            Release_min=Release_min,
            Release_max=Release_max,
            Release_alpha1=Release_alpha1,
            Release_alpha2=Release_alpha2,
            Release_beta1=Release_beta1,
            Release_beta2=Release_beta2,
            Release_p1=Release_p1,
            Release_p2=Release_p2,
            Release_c=Release_c,
            GRanD_CAP_MCM=GRanD_CAP_MCM,
            Obs_MEANFLOW_CUMECS=Obs_MEANFLOW_CUMECS,
            calc_method=calc_method,
            io_in_cfs=io_in_cfs,
            compute_daily=False,
            nhrs_substep=nhrs_substep,
            budget_type=budget_type,
        )
        self.name = "StarfitSourceSinkFlowNode"

        # remaining code from source_sink_flow_node
        self._source_sink_storage_min = source_sink_storage_min
        self._source_sink_data = source_sink_data
        self._missing_data_as_zero = missing_data_as_zero

        first_time = source_sink_data.index[0]
        if not isinstance(first_time, pd.Timestamp):
            try:
                _ = datetime.strptime(first_time, "%Y-%m-%d")
            except ValueError as err_msg:
                msg = (
                    f"The source_sink_data index Date column is malformed: "
                    f"{err_msg}"
                )
                raise ValueError(msg)

        return

    @staticmethod
    def get_mass_budget_terms():
        """Get a dictionary of variable names for mass budget terms."""
        return {
            "inputs": [
                "_lake_inflow",
            ],
            "outputs": [
                "_lake_release",
                "_lake_spill",
            ],
            "storage_changes": [
                "_lake_storage_change_flow_units",
                "_negative_sink_source",
            ],
        }

    @property
    def sink_source(self) -> np.float64:
        return self._sink_source[0]

    def _prepare_timestep_hourly(self) -> None:
        super()._prepare_timestep_hourly()
        self._prepare_timestep_source_sink()
        return

    def _prepare_timestep_source_sink(self) -> None:
        self._sink_source_sum = zero

        # remaining code from source_sink_flow_node
        ymd = self.control.current_datetime.strftime("%Y-%m-%d")
        if ymd not in self._source_sink_data.index:
            if not self._missing_data_as_zero:
                msg = (
                    "Missing data not handled by SourceSinkFlowNode unless "
                    "missing_data_as_zero set to True."
                )
                raise ValueError(msg)
            # <
            self._source_sink_requested = zero
        else:
            self._source_sink_requested = self._source_sink_data[ymd]
            if self._io_in_cfs:
                self._source_sink_requested *= cfs_to_cms

        return

    def finalize_timestep(self):
        self._negative_sink_source[:] = -1 * self._sink_source
        super().finalize_timestep()

    def _source_sink_calculations(
        self,
        isubstep: int,
        flow_to_vol_conversion: float,
    ) -> None:
        """Calculate flow diversion from storage on the time basis supplied.

        Note units are flow units

        Args:
            flow_to_vol_conversion: This conversion is from cubic meters per
              second to millions of cubic meteres (MCM) for some period of
              time. If the calculation is hourly, the conversion would be the
              pywatershed constant m3ps_to_MCM_hourly. Conversions could be
              made for other periods of time.
        """
        source_sink_vol = self._source_sink_requested * flow_to_vol_conversion
        min_storage = self._source_sink_storage_min
        storage = self._lake_storage_sub.copy()  # MCM

        if source_sink_vol >= zero:
            # a source is always applied
            storage += source_sink_vol

        elif (source_sink_vol < zero) and (storage < min_storage):
            # sink is not applied when storage < min_storage
            # storage stays the same
            source_sink_vol = zero

        elif (source_sink_vol < zero) and (storage >= min_storage):
            if (storage + source_sink_vol) < min_storage:
                # if the sink is too much, reduce it up to min storage
                # The difference here is to get the negative sign
                source_sink_vol = min_storage - storage
                storage = min_storage
            else:
                storage += source_sink_vol

        # <
        self._lake_storage_after_source_sink[:] = storage  # MCM
        self._source_sink[:] = source_sink_vol * (one / flow_to_vol_conversion)

        # very confusing, move self._source_sink to self._sink_source_sub
        self._sink_source_sum += self._source_sink[0]
        self._sink_source[:] = self._sink_source_sum / (isubstep + 1)
        # conversion to cfs in self._release_calculations_post_hourly
        return

    def _calc_storage_change_sub_hourly(self) -> None:
        # spill is not included in this calculation
        self._lake_storage_change_sub[:] = (
            self._lake_inflow_sub - self._lake_release_sub + self._source_sink
        ) * self._m3ps_to_MCM  # MCM: million cubic meters
        return

    def _calc_subtimestep_hourly(
        self, isubstep, inflow_upstream, inflow_lateral
    ) -> None:
        self._pre_release_calculations_hourly(inflow_upstream, inflow_lateral)
        # given the existing storage, a storage minimum and a requested
        # diversion, calculate the diversion and the reesulting storage
        self._source_sink_calculations(
            isubstep=isubstep, flow_to_vol_conversion=self._m3ps_to_MCM
        )
        (
            self._lake_release_sub[:],
            self._lake_availability_status_sub[:],
        ) = Starfit._calc_istarf_release(
            epiweek=np.minimum(self.control.current_epiweek, 52),
            GRanD_CAP_MCM=self._GRanD_CAP_MCM,
            grand_id=self._grand_id,
            lake_inflow=self._lake_inflow_sub,
            lake_storage=self._lake_storage_after_source_sink,
            NORhi_alpha=self._NORhi_alpha,
            NORhi_beta=self._NORhi_beta,
            NORhi_max=self._NORhi_max,
            NORhi_min=self._NORhi_min,
            NORhi_mu=self._NORhi_mu,
            NORlo_alpha=self._NORlo_alpha,
            NORlo_beta=self._NORlo_beta,
            NORlo_max=self._NORlo_max,
            NORlo_min=self._NORlo_min,
            NORlo_mu=self._NORlo_mu,
            Obs_MEANFLOW_CUMECS=self._Obs_MEANFLOW_CUMECS,
            Release_alpha1=self._Release_alpha1,
            Release_alpha2=self._Release_alpha2,
            Release_beta1=self._Release_beta1,
            Release_beta2=self._Release_beta2,
            Release_c=self._Release_c,
            Release_max=self._Release_max,
            Release_min=self._Release_min,
            Release_p1=self._Release_p1,
            Release_p2=self._Release_p2,
        )  # output in m^3/d

        self._post_release_calculations_hourly(isubstep)
        return

    def _post_release_calculations_hourly(self, isubstep) -> None:
        super()._post_release_calculations_hourly(isubstep)
        if self._io_in_cfs:
            self._sink_source[:] *= cms_to_cfs
        return


class StarfitSourceSinkFlowNodeMaker(StarfitFlowNodeMaker):
    r"""STARFIT SourceSink FlowNodeMaker: STARFIT with diversions on storage."""  # noqa: E501

    def __init__(
        self,
        discretization: Parameters,
        parameters: Parameters,
        source_sink_df: pd.DataFrame,
        missing_data_as_zero: bool = False,
        calc_method: Literal["numba", "numpy"] = None,
        io_in_cfs: bool = True,
        verbose: bool = None,
        budget_type: Literal["defer", None, "warn", "error"] = None,
        nhrs_substep: int = 1,
    ) -> None:
        """Instantiate StarfitFlowNodeMaker.

        Args:
            discretization: A :class:`Parameters` object.
            parameters: A :class:`Parmaeters` object with the parameters listed
                in :class:`StarfitFlowNode` dimensioned by number of
                reservoirs/nodes to instantiate.
            source_sink_df: A pandas DataFrame of observations with a time
                index which can be selected by '%Y-%m-%d' strftime of a
                datetime64. The column names are not used and may be anything,
                for example the nhm_seg of the upstream segment. However, the
                columns order MUST be collated with the input data vectors. The
                sign convention is: sources are positive and sinks are
                negative. That is, the sign is from the perspective of the
                node. Units are cubic feet per second.
            missing_data_as_zero: Bool option to treat missing times in the
                timeseries as having zero source/sink.
            calc_method: One of "numba" or "numpy".
            io_in_cfs: Are the units in cubic feet per second? False gives
                units of cubic meters per second.
            nhrs_substep: Number of hours in the subtimestep.
            budget_type: One of "defer", "warn", or "error".
            verbose: bool = None,
        """
        super().__init__(
            discretization=discretization,
            parameters=parameters,
            calc_method=calc_method,
            io_in_cfs=io_in_cfs,
            verbose=verbose,
            budget_type=budget_type,
            nhrs_substep=nhrs_substep,
        )
        self.name = "StarfitSourceSinkNodeMaker"
        self._source_sink_df = source_sink_df
        self._missing_data_as_zero = missing_data_as_zero

    def get_node(
        self, control: Control, index: int
    ) -> StarfitSourceSinkFlowNode:
        stor_min = self._parameters.parameters["source_sink_storage_min"][
            index
        ]
        col_name = self._source_sink_df.columns[index]
        data_ts = pd.Series(self._source_sink_df[col_name])
        return StarfitSourceSinkFlowNode(
            control=control,
            grand_id=self.grand_id[index],
            initial_storage=self.initial_storage[index],
            start_time=self.start_time[index],
            end_time=self.end_time[index],
            inflow_mean=self.inflow_mean[index],
            NORhi_min=self.NORhi_min[index],
            NORhi_max=self.NORhi_max[index],
            NORhi_alpha=self.NORhi_alpha[index],
            NORhi_beta=self.NORhi_beta[index],
            NORhi_mu=self.NORhi_mu[index],
            NORlo_min=self.NORlo_min[index],
            NORlo_max=self.NORlo_max[index],
            NORlo_alpha=self.NORlo_alpha[index],
            NORlo_beta=self.NORlo_beta[index],
            NORlo_mu=self.NORlo_mu[index],
            Release_min=self.Release_min[index],
            Release_max=self.Release_max[index],
            Release_alpha1=self.Release_alpha1[index],
            Release_alpha2=self.Release_alpha2[index],
            Release_beta1=self.Release_beta1[index],
            Release_beta2=self.Release_beta2[index],
            Release_p1=self.Release_p1[index],
            Release_p2=self.Release_p2[index],
            Release_c=self.Release_c[index],
            GRanD_CAP_MCM=self.GRanD_CAP_MCM[index],
            Obs_MEANFLOW_CUMECS=self.Obs_MEANFLOW_CUMECS[index],
            source_sink_storage_min=stor_min,
            source_sink_data=data_ts,
            missing_data_as_zero=self._missing_data_as_zero,
            calc_method=self._calc_method,
            io_in_cfs=self._io_in_cfs,
            budget_type=self._budget_type,
            nhrs_substep=self._nhrs_substep,
        )

    @staticmethod
    def get_dimensions() -> tuple:
        """Get a tuple of dimension names for this StarfitFlowNodeMaker."""
        return ("nreservoirs",)

    @staticmethod
    def get_parameters() -> tuple:
        """Get a tuple of parameter names for StarfitFlowNodeMaker."""
        return (
            "grand_id",
            "initial_storage",
            "start_time",
            "end_time",
            "inflow_mean",
            "NORhi_min",
            "NORhi_max",
            "NORhi_alpha",
            "NORhi_beta",
            "NORhi_mu",
            "NORlo_min",
            "NORlo_max",
            "NORlo_alpha",
            "NORlo_beta",
            "NORlo_mu",
            "Release_min",
            "Release_max",
            "Release_alpha1",
            "Release_alpha2",
            "Release_beta1",
            "Release_beta2",
            "Release_p1",
            "Release_p2",
            "Release_c",
            "GRanD_CAP_MCM",
            "Obs_MEANFLOW_CUMECS",
            "source_sink_storage_min",
        )
