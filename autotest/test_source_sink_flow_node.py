import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatershed import PRMSChannel
from pywatershed.base.adapter import Adapter, AdapterNetcdf, adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.flow_graph import FlowGraph
from pywatershed.base.parameters import Parameters
from pywatershed.constants import nan, zero
from pywatershed.hydrology.prms_channel_flow_graph import (
    HruSegmentFlowAdapter,
    PRMSChannelFlowNodeMaker,
)
from pywatershed.hydrology.source_sink_flow_node import SourceSinkFlowNodeMaker
from pywatershed.parameters import PrmsParameters

do_compare_output_files = False
do_compare_in_memory = True
rtol = atol = 1.0e-7

rename_vars = {
    "channel_outflow_vol": "outflows",
    "seg_upstream_inflow": "node_upstream_inflows",
    "seg_outflow": "node_outflows",
    "seg_stor_change": "node_storage_changes",
}

# the above values in rename_vars need converted to volume in these cases
convert_to_vol = ["outflows", "node_storage_changes"]


@pytest.fixture(scope="function")
def control(simulation):
    # JLM TODO: _obsin restriction on control_pattern?
    if "ucb_2yr:nhm" not in simulation["name"]:
        pytest.skip("Only testing obsin flow graph for ucb_2yr:nhm")
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    del control.options["netcdf_output_dir"]
    control.edit_n_time_steps(180)
    return control


@pytest.fixture(scope="function")
def discretization_prms(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    dis_seg_file = simulation["dir"] / "parameters_dis_seg.nc"
    return Parameters.merge(
        Parameters.from_netcdf(dis_hru_file, encoding=False),
        Parameters.from_netcdf(dis_seg_file, encoding=False),
    )


@pytest.fixture(scope="function")
def parameters_prms(simulation, control):
    param_file = simulation["dir"] / "parameters_PRMSChannel.nc"
    return PrmsParameters.from_netcdf(param_file)


@pytest.fixture(scope="function")
def source_sink_data(simulation, parameters_prms):
    # bring in the source_sink data from csv file
    sink_file = simulation["dir"] / "UCB_sinks_notreal.csv"
    source_sink_data = pd.read_csv(sink_file)
    source_sink_data = source_sink_data.set_index("time")
    old_cols = source_sink_data.columns
    new_cols = source_sink_data.columns.astype(np.int64)
    source_sink_data.rename(
        columns=dict(zip(old_cols, new_cols)), inplace=True
    )

    return source_sink_data


@pytest.mark.parametrize(
    "case",
    [
        "base",
        "all-zero",
        "missing-fail",
        "missing-as-zero",
    ],
)
def test_prms_channel_obsin_compare_prms(
    simulation,
    control,
    discretization_prms,
    parameters_prms,
    source_sink_data,
    tmp_path,
    case,
):
    # Set the parameter(s)
    sink_nhm_seg = source_sink_data.columns
    # this loop keeps sink_nhm_seg and sink_inds collated.
    sink_inds = []
    for ss in sink_nhm_seg:
        sink_inds += np.where(parameters_prms.parameters["nhm_seg"] == ss)[
            0
        ].tolist()

    source_sink_data = source_sink_data[
        source_sink_data.columns.intersection(sink_nhm_seg)
    ]
    nsink = len(sink_inds)

    sink_params_ds = (
        discretization_prms.subset(["nhm_seg"])
        .to_xr_ds()
        .isel(nsegment=sink_inds)
    )
    sink_params_ds["flow_min"] = xr.Variable(
        "nsegment", np.array([0.00, 0.01])
    )
    # nhm_seg is not a parameter of the method, but the names can be kept
    # and will be ignored because the position of the columns is used.
    sink_params = Parameters.from_ds(sink_params_ds.drop_vars("nhm_seg"))

    # combine PRMS lateral inflows to a single non-volumetric inflow
    output_dir = simulation["output_dir"]
    input_variables = {}
    for key in PRMSChannel.get_inputs():
        nc_path = output_dir / f"{key}.nc"
        input_variables[key] = AdapterNetcdf(nc_path, key, control)

    inflows_prms = HruSegmentFlowAdapter(parameters_prms, **input_variables)

    class GraphInflowAdapter(Adapter):
        def __init__(
            self,
            prms_inflows: Adapter,
            nsink: int,
            variable: str = "inflows",
        ):
            self._variable = variable
            self._prms_inflows = prms_inflows

            self._nsink = nsink
            self._nnodes = len(self._prms_inflows.current) + nsink
            self._current_value = np.zeros(self._nnodes) * nan
            return

        def advance(self) -> None:
            self._prms_inflows.advance()
            self._current_value[0:-nsink] = self._prms_inflows.current
            self._current_value[-nsink:] = zero  # no inflow at the obsin
            return

    inflows_graph = GraphInflowAdapter(inflows_prms, nsink)

    # FlowGraph
    missing_data_as_zero = False
    if case == "base":
        pass
    elif case == "all-zero":
        source_sink_data *= 0.0

    elif case == "missing-as-zero":
        missing_data_as_zero = True
        source_sink_data = source_sink_data.loc[
            (source_sink_data != 0).any(axis=1)
        ]

    elif case == "missing-fail":
        source_sink_data = source_sink_data.loc[
            (source_sink_data != 0).any(axis=1)
        ]

    node_maker_dict = {
        "prms_channel": PRMSChannelFlowNodeMaker(
            discretization_prms, parameters_prms
        ),
        "sink": SourceSinkFlowNodeMaker(
            sink_params, source_sink_data, missing_data_as_zero
        ),
    }
    nnodes = parameters_prms.dims["nsegment"] + nsink
    node_maker_name = ["prms_channel"] * nnodes
    node_maker_name[-nsink:] = ["sink"] * nsink
    node_maker_index = np.arange(nnodes)
    node_maker_index[-nsink:] = np.arange(nsink)
    node_maker_id = np.arange(nnodes)
    to_graph_index = np.zeros(nnodes, dtype=np.int64)
    dis_params = discretization_prms.parameters
    to_graph_index[0:-nsink] = dis_params["tosegment"] - 1

    wh_intervene_below_nhm = sink_inds
    wh_intervene_above_nhm = (dis_params["tosegment"] - 1)[
        wh_intervene_below_nhm
    ]

    # have to map to the graph from an index found in prms_channel
    # in a way that preserves the order in sink_ind
    wiag = []
    wibg = []
    for ii in range(nsink):
        wiag += np.where(
            (np.array(node_maker_name) == "prms_channel")
            & (node_maker_index == wh_intervene_above_nhm[ii])
        )[0].tolist()

        wibg += np.where(
            (np.array(node_maker_name) == "prms_channel")
            & (node_maker_index == wh_intervene_below_nhm[ii])
        )[0].tolist()

    wh_intervene_above_graph = wiag
    wh_intervene_below_graph = wibg

    to_graph_index[-nsink:] = wh_intervene_above_graph
    to_graph_index[wh_intervene_below_graph] = np.arange(nsink) + (
        nnodes - nsink
    )

    params_flow_graph = Parameters(
        dims={
            "nnodes": nnodes,
        },
        coords={
            "node_coord": np.arange(nnodes),
        },
        data_vars={
            "node_maker_name": np.array(node_maker_name),
            "node_maker_index": node_maker_index,
            "node_maker_id": node_maker_id,
            "to_graph_index": to_graph_index,
        },
        metadata={
            "node_coord": {"dims": ["nnodes"]},
            "node_maker_name": {"dims": ["nnodes"]},
            "node_maker_index": {"dims": ["nnodes"]},
            "node_maker_id": {"dims": ["nnodes"]},
            "to_graph_index": {"dims": ["nnodes"]},
        },
        validate=True,
    )

    control.options["netcdf_output_var_names"] = ["sink_source"]
    flow_graph = FlowGraph(
        control,
        discretization=None,
        parameters=params_flow_graph,
        inflows=inflows_graph,
        node_maker_dict=node_maker_dict,
        imbalance_behavior="error",
        addtl_output_vars=["sink_source"],
    )

    # if do_compare_output_files:

    nc_parent = tmp_path / simulation["name"].replace(":", "_")
    flow_graph.initialize_netcdf(nc_parent)

    if do_compare_in_memory:
        answers = {}
        for var in PRMSChannel.get_variables():
            if var not in rename_vars.keys():
                continue
            new_var = rename_vars[var]
            var_pth = output_dir / f"{var}.nc"
            answers[new_var] = adapter_factory(
                var_pth, variable_name=var, control=control
            )

    if not case == "all-zero":
        # the sink inds are the indices of the nhm_segments just above
        # the diversions. we want to ignore all the indices below that.
        wh_ignore = []
        for si in sink_inds:
            upind = si
            toseg = discretization_prms.parameters["tosegment"] - 1
            while upind >= 0:
                downind = toseg[upind]
                wh_ignore += [upind]
                upind = downind
        wh_ignore = np.unique(wh_ignore)
        # also get the downstream inds for check that inflows are altered
        # by diversions at those locations
        downstream_reach_inds = [toseg[si] for si in sink_inds]

    if case == "missing-fail":
        match = (
            "Missing data not handled by SourceSinkFlowNode unless "
            "missing_data_as_zero set to True."
        )
        with pytest.raises(ValueError, match=match):
            control.advance()
            flow_graph.advance()
            flow_graph.calculate(1.0)

        return

    for istep in range(control.n_times):
        control.advance()
        flow_graph.advance()
        flow_graph.calculate(1.0)
        flow_graph.output()

        if do_compare_in_memory:
            for key, val in answers.items():
                val.advance()
                if key in convert_to_vol:
                    desired = val.current / (24 * 60 * 60)
                else:
                    desired = val.current

                actual = flow_graph[key][0:-nsink]
                if not case == "all-zero":
                    # check that the inflows are altered by the data
                    if key == "node_upstream_inflows":
                        # this is an ad hoc version of the algorithm
                        for ii, ds in enumerate(downstream_reach_inds):
                            col_key = parameters_prms.parameters["nhm_seg"][
                                sink_inds[ii]
                            ]
                            dd = desired[ds]
                            ymd = control.current_datetime.strftime("%Y-%m-%d")
                            if ymd in source_sink_data[col_key].index:
                                req_ss = source_sink_data[col_key][ymd]
                            else:
                                req_ss = zero
                            if req_ss < 0:
                                flo_min = sink_params_ds["flow_min"][ii].values
                                aa = dd + req_ss
                                if aa < flo_min:
                                    aa = flo_min
                            else:
                                aa = dd + req_ss

                            assert abs(aa - actual[ds]) < 1.0e-7

                    # and otherwise ignore all the reaches downstream of
                    # the diversions
                    desired[wh_ignore] = actual[wh_ignore]

                np.testing.assert_allclose(
                    actual,
                    desired,
                    atol=atol,
                    rtol=rtol,
                )

    flow_graph.finalize()
