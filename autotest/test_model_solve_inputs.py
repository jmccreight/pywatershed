import pytest

import pywatershed as pws

NHM_PROCESSES = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphere,
    pws.PRMSCanopy,
    pws.PRMSSnow,
    pws.PRMSRunoff,
    pws.PRMSSoilzone,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]


@pytest.mark.domainless
def test_solve_inputs_nhm_process_list():
    """The full NHM: only the forcings come from file."""
    result = pws.Model.solve_inputs(NHM_PROCESSES)

    assert result["from_file"] == ["prcp", "tmax", "tmin"]

    inputs_from = result["inputs_from"]
    # spot-check process-to-process wiring
    assert inputs_from["PRMSAtmosphere"]["soltab_potsw"] == (
        "PRMSSolarGeometry"
    )
    assert inputs_from["PRMSAtmosphere"]["prcp"] is None
    assert inputs_from["PRMSSnow"]["net_ppt"] == "PRMSCanopy"
    assert inputs_from["PRMSChannel"]["sroff_vol"] == "PRMSRunoff"
    assert inputs_from["PRMSChannel"]["gwres_flow_vol"] == "PRMSGroundwater"
    # a process with no inputs has an empty entry
    assert inputs_from["PRMSSolarGeometry"] == {}


@pytest.mark.domainless
def test_solve_inputs_below_snow_subset():
    """A truncated model: everything the missing processes supplied is
    needed from file (e.g. forcing a sub-model from another model's
    output)."""
    subset = [
        pws.PRMSRunoffAg,
        pws.PRMSSoilzoneAg,
        pws.PRMSGroundwater,
        pws.PRMSChannel,
    ]
    result = pws.Model.solve_inputs(subset)

    # atmosphere/canopy/snow-supplied inputs are now file inputs
    for input_name in [
        "potet",
        "transp_on",
        "net_ppt",
        "net_rain",
        "net_snow",
        "snowmelt",
        "snowcov_area",
        "snow_evap",
        "hru_intcpevap",
    ]:
        assert input_name in result["from_file"]

    # within-subset wiring is unaffected
    inputs_from = result["inputs_from"]
    assert inputs_from["PRMSSoilzoneAg"]["infil_ag"] == "PRMSRunoffAg"
    assert inputs_from["PRMSChannel"]["sroff_vol"] == "PRMSRunoffAg"


@pytest.mark.domainless
def test_solve_inputs_model_dict_passed_inputs():
    """Dict form: a spec key naming an input in the class's __init__
    signature counts as passed and drops out of the wiring; non-process
    entries are ignored."""
    model_dict = {
        "control": "not a process spec",
        "dis_hru": "not a process spec",
        "runoff": {"class": pws.PRMSRunoffAg},
        "soilzone": {"class": pws.PRMSSoilzoneAg, "ag_frac": "a value"},
    }
    result = pws.Model.solve_inputs(model_dict)

    assert set(result["inputs_from"].keys()) == {"runoff", "soilzone"}
    # passed for soilzone: omitted from its wiring...
    assert "ag_frac" not in result["inputs_from"]["soilzone"]
    # ...but runoff also consumes ag_frac and was not passed it, so it
    # remains a file input (passing is per process spec, not global)
    assert result["inputs_from"]["runoff"]["ag_frac"] is None
    assert "ag_frac" in result["from_file"]

    # with only the passed consumer present, it drops out entirely
    del model_dict["runoff"]
    result = pws.Model.solve_inputs(model_dict)
    assert "ag_frac" not in result["from_file"]

    # and without passing it, it is a file input
    model_dict["soilzone"] = {"class": pws.PRMSSoilzoneAg}
    result = pws.Model.solve_inputs(model_dict)
    assert result["inputs_from"]["soilzone"]["ag_frac"] is None
    assert "ag_frac" in result["from_file"]
