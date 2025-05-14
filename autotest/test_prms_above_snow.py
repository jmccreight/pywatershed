import pathlib as pl
import shutil
from pprint import pprint

import pytest
from utils_compare import compare_model_in_memory

import pywatershed
from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.model import Model
from pywatershed.parameters import Parameters, PrmsParameters

# This test removes PRMSSnow from the full model test, removing its errorrs.
# Importantly, it tests that "pptmix" is computed correctly as it is set by
# both PRMSAtmosphere and PRMSCanopy.
# The (nearly) comprehensive suite of variables is verified against PRMS
# outputs. Several variables in soilzone are not output in the PRMS
# configuraton or perhaps not at all by PRMS and these are skipped.
# The (nearly) comprehensive suite of variables is verified against PRMS
# outputs. Several variables in soilzone are not output in the PRMS
# configuraton or perhaps not at all by PRMS and these are skipped.
# Importantly, this test and test_prms_below_snow test model
# instantiation/invocation three ways which is not tested elsewhere in the
# the test suite.

invoke_style = ("prms", "model_dict", "model_dict_from_yaml")
fail_after_all_vars = False
verbosity = 0

all_configs_same = [
    pywatershed.PRMSSolarGeometry,
    pywatershed.PRMSAtmosphere,
    pywatershed.PRMSCanopy,
]

test_models = {
    "nhm": all_configs_same,
    "nhm_no_dprst": all_configs_same,
    "sagehen_no_cascades": all_configs_same,
    "sagehen_gridded_cascades": all_configs_same,
    "sagehen_structured_example": all_configs_same,
}

comparison_vars_dict_all = {
    "PRMSSolarGeometry": pywatershed.PRMSSolarGeometry.get_variables(),
    "PRMSAtmosphere": pywatershed.PRMSAtmosphere.get_variables(),
    "PRMSCanopy": list(
        set(pywatershed.PRMSCanopy.get_variables()) - {"intcp_transp_on"}
    ),
}

tol = {
    "PRMSSolarGeometry": 1.0e-8,
    "PRMSAtmosphere": 1.0e-5,
    "PRMSCanopy": 1.0e-6,
}


@pytest.fixture(scope="function")
def control(simulation):
    sim_name = simulation["name"]
    config_name = sim_name.split(":")[1]
    if config_name not in test_models.keys():
        pytest.skip(
            f"The configuration is not tested by test_model: {config_name}"
        )
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.options["verbosity"] = 10
    control.options["budget_type"] = None
    control.options["calc_method"] = "numba"
    del control.options["netcdf_output_var_names"]
    return control


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    dis_hru = Parameters.from_netcdf(dis_hru_file, encoding=False)
    dis = {"dis_hru": dis_hru}

    return dis


@pytest.fixture(scope="function", params=invoke_style)
def model_args(simulation, control, discretization, request):
    invoke_style = request.param
    control_key = simulation["name"].split(":")[1]
    process_list = test_models[control_key]

    if invoke_style == "prms":
        # Single params is the backwards compatible way
        param_file = simulation["dir"] / control.options["parameter_file"]
        args = {
            "process_list_or_model_dict": process_list,
            "control": control,
            "parameters": PrmsParameters.load(param_file),
        }

    elif invoke_style == "model_dict":
        # Constructing this model_dict is the new way
        model_dict = discretization
        model_dict["control"] = control
        # could use any names
        model_dict["model_order"] = [
            pp.__name__.lower() for pp in process_list
        ]

        for process in test_models[control_key]:
            proc_name = process.__name__
            proc_name_lower = proc_name.lower()
            model_dict[proc_name_lower] = {}
            proc = model_dict[proc_name_lower]
            proc["class"] = process
            proc_param_file = simulation["dir"] / f"parameters_{proc_name}.nc"
            proc["parameters"] = PrmsParameters.from_netcdf(proc_param_file)
            proc["dis"] = "dis_hru"

        if verbosity > 0:
            pprint(model_dict, sort_dicts=False)

        args = {
            "process_list_or_model_dict": model_dict,
            "control": None,
            "parameters": None,
        }

    elif invoke_style == "model_dict_from_yaml":
        # Note this invocation ignores control and parameters
        yaml_name = simulation["name"].split(":")[1]
        yaml_file = simulation["dir"] / f"{yaml_name}_model.yaml"
        model_dict = Model.model_dict_from_yaml(yaml_file)

        # Edit this dict from the yaml to use only processes below snow
        class_keys = {
            vv["class"]: kk
            for kk, vv in model_dict.items()
            if isinstance(vv, dict) and "class" in vv.keys()
        }
        to_del = [
            kk for cl, kk in class_keys.items() if cl not in process_list
        ]
        for del_me in to_del:
            _ = model_dict["model_order"].remove(del_me)
            del model_dict[del_me]

        args = {
            "process_list_or_model_dict": model_dict,
            "control": None,
            "parameters": None,
        }

    else:
        msg = "invalid parameter value"
        raise ValueError(msg)

    return args


def test_model(simulation, model_args, tmp_path):
    """Run the model"""
    tmp_path = pl.Path(tmp_path)
    output_dir = simulation["output_dir"]
    sim_name = simulation["name"]
    config_name = sim_name.split(":")[1]

    # setup input_dir with symlinked prms inputs and outputs
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for ff in output_dir.resolve().glob("*.nc"):
        shutil.copy(ff, input_dir / ff.name)
    for ff in output_dir.parent.resolve().glob("*.nc"):
        shutil.copy(ff, input_dir / ff.name)

    if model_args["control"] is None:
        control = model_args["process_list_or_model_dict"]["control"]
    else:
        control = model_args["control"]

    control.options["input_dir"] = input_dir
    model_out_dir = tmp_path / "output"
    control.options["netcdf_output_dir"] = model_out_dir

    model = Model(**model_args, write_control=model_out_dir)

    # check that control yaml file was written
    control_yaml_file = sorted(model_out_dir.glob("*model_control.yaml"))
    assert len(control_yaml_file) == 1

    # ---------------------------------
    # get the answer data against PRMS5.2.1
    # this is the adhoc set of things to compare, to circumvent fussy issues?

    comparison_vars_dict = {}

    plomd = model_args["process_list_or_model_dict"]
    config_processes = test_models[config_name]
    if isinstance(plomd, list):
        processes = [pp for pp in plomd if pp in config_processes]
        control = model_args["control"]
        # class_key = {
        #     vv.__class__.__name__: kk for kk, vv in model.processes.items()
        # }
    else:
        processes = [
            vv["class"]
            for vv in plomd.values()
            if isinstance(vv, dict) and vv["class"] in config_processes
        ]
        processes = [pp for pp in processes if pp in config_processes]
        control = plomd["control"]
        # class_key = {
        #     vv["class"].__name__: kk
        #     for kk, vv in plomd.items()
        #     if isinstance(vv, dict) and "class" in vv.keys()
        # }

    for cls in processes:
        key = cls.__name__
        cls_vars = cls.get_variables()
        comparison_vars_dict[key] = {
            vv for vv in comparison_vars_dict_all[key] if vv in cls_vars
        }

    # Read PRMS output into ans for comparison with pywatershed results
    ans = {key: {} for key in comparison_vars_dict.keys()}
    for process_name, var_names in comparison_vars_dict.items():
        for vv in var_names:
            # TODO: this is hacky, improve the design
            if (
                "dprst_flag" in control.options.keys()
                and not control.options["dprst_flag"]
            ):
                if "dprst" in vv:
                    continue

            if vv in ["tmin", "tmax", "prcp"]:
                nc_pth = input_dir.parent / f"{vv}.nc"
            else:
                nc_pth = input_dir / f"{vv}.nc"

            ans[process_name][vv] = adapter_factory(
                nc_pth, variable_name=vv, control=control
            )

    for istep in range(control.n_times):
        model.advance()
        model.calculate()

        _ = compare_model_in_memory(
            model=model,
            answers=ans,
            rtol=tol,
            atol=tol,
            fail_after_all_vars=fail_after_all_vars,
            skip_missing_ans=True,
            verbosity=verbosity,
        )

    # <
    model.finalize()

    return
