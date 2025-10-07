<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [PRMSSoilzoneAg Implementation Summary](#prmssoilzoneag-implementation-summary)
  - [Overview](#overview)
  - [Source Reference](#source-reference)
  - [Key Features Implemented](#key-features-implemented)
    - [1. Dual Area Soil Moisture Accounting](#1-dual-area-soil-moisture-accounting)
    - [2. Iterative AET Matching](#2-iterative-aet-matching)
    - [3. Agricultural Parameters](#3-agricultural-parameters)
    - [4. Agricultural State Variables](#4-agricultural-state-variables)
    - [5. Inputs](#5-inputs)
  - [Simplifications (Features Skipped)](#simplifications-features-skipped)
    - [1. GLACIER HRU Type](#1-glacier-hru-type)
    - [2. GSFLOW Integration (GSFLOW_flag==ACTIVE)](#2-gsflow-integration-gsflow_flagactive)
    - [3. Cascade Flow (Cascade_flag > CASCADE_OFF)](#3-cascade-flow-cascade_flag--cascade_off)
    - [4. Swale HRUs (compute_lateral==OFF)](#4-swale-hrus-compute_lateraloff)
    - [5. Basin Aggregations](#5-basin-aggregations)
    - [6. MODSIM Integration](#6-modsim-integration)
    - [7. Frozen Ground (CFGI)](#7-frozen-ground-cfgi)
  - [Code Organization](#code-organization)
    - [Class Structure](#class-structure)
    - [Calculation Flow](#calculation-flow)
  - [Key Fortran-to-Python Mappings](#key-fortran-to-python-mappings)
  - [Mass Budget](#mass-budget)
  - [Testing Recommendations](#testing-recommendations)
  - [Known Limitations](#known-limitations)
  - [Future Enhancements](#future-enhancements)
  - [Usage Example](#usage-example)
  - [Contact](#contact)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# PRMSSoilzoneAg Implementation Summary

## Overview

This document summarizes the implementation of `PRMSSoilzoneAg` in `prms_soilzone_ag.py`, which ports the GSFLOW 2.4.0 `soilzone_ag.f90` Fortran code to Python.

## Source Reference

- **Fortran Source**: `pywatershed/gsflow_src/2.4.0/src/soilzone_ag.f90`
- **Python Implementation**: `pywatershed/pywatershed/hydrology/prms_soilzone_ag.py`
- **Base Class**: Inherits from `ConservativeProcess` (like `PRMSSoilzone`)

## Key Features Implemented

### 1. Dual Area Soil Moisture Accounting

Each HRU is divided into two areas, each with independent soil moisture accounting:

- **Pervious Area**: Traditional PRMS soil zone (same as base `PRMSSoilzone`)
- **Agricultural/Irrigated Area**: Separate soil moisture states for agricultural land

**Fortran Reference**: Lines ~752-767 in `szrun_ag()`

### 2. Iterative AET Matching

The core innovation: iteratively adjusts irrigation to match observed actual ET.

**Algorithm**:
1. Start with initial infiltration + any irrigation estimates
2. Compute agricultural ET
3. Compare to observed AET (from `AET_external` input)
4. If `unsatisfied_ag_et > soilzone_aet_converge`:
   - Add estimated irrigation: `ag_irrigation_add += unsatisfied_ag_et`
   - Repeat from step 1
5. Converge when all HRUs satisfy tolerance or max iterations reached

**Parameters**:
- `max_soilzone_ag_iter`: Maximum iterations (default: 10)
- `soilzone_aet_converge`: Convergence tolerance (default: 0.00001 inches)
- `ag_soilwater_deficit_min`: Minimum deficit to trigger irrigation

**Fortran Reference**: 
- Main iteration loop: Lines ~442-1187 `DO WHILE (keep_iterating==ACTIVE)`
- Irrigation adjustment: Lines ~1127-1160

### 3. Agricultural Parameters

New parameters specific to agricultural areas:

- `ag_soil_type`: Soil type for ag area (1=sand, 2=loam, 3=clay)
- `ag_soil_moist_max`, `ag_soil_rechr_max`: Maximum storage capacities
- `ag_cov_type`: Vegetation cover type for ag area
- `ag_covden_sum`, `ag_covden_win`: Cover density (summer/winter)
- `ag_soil2gw_max`: Maximum flow from ag soil to groundwater
- `ag_soilwater_deficit_min`: Irrigation trigger threshold

**Fortran Reference**: Lines ~93-264 in `szdecl_ag()`

### 4. Agricultural State Variables

Separate state variables for agricultural area:

- `ag_soil_moist`: Soil moisture in ag area (inches)
- `ag_soil_rechr`: Recharge zone storage in ag area (inches)
- `ag_soil_lower`: Lower zone storage in ag area (inches)
- `ag_actet`: Actual ET from ag area (inches)
- `ag_irrigation_add`: Estimated irrigation added (inches over ag area)
- `ag_soil_to_gw`: Direct recharge from ag soil to groundwater (inches)
- `ag_soil_to_gvr`: Excess ag soil water to gravity reservoir (inches)

**Fortran Reference**: Lines ~38-62 in module declarations

### 5. Inputs

New inputs compared to base `PRMSSoilzone`:

- `infil_ag`: Infiltration to agricultural area (separate from `infil_hru`)
- `ag_frac`: Fraction of HRU that is agricultural (0-1)
- `AET_external`: Observed actual ET for each HRU (when `iter_aet_flag=True`)

**Fortran Reference**: Lines ~373-385 in `szrun_ag()`

## Simplifications (Features Skipped)

Per user request, the following features from the Fortran code were **not** implemented:

### 1. GLACIER HRU Type
- **Fortran**: Scattered checks for `Hru_type(ihru)==GLACIER`
- **Reason**: Not needed for current applications

### 2. GSFLOW Integration (GSFLOW_flag==ACTIVE)
- **Fortran**: Lines with `IF ( GSFLOW_flag==ACTIVE ) THEN`
- **Features skipped**:
  - Gravity reservoir interaction with MODFLOW
  - `compute_gravflow_ag()` subroutine
  - `Gvr2sm`, `Gvr2ag` (groundwater to soil moisture flow)
  - `Gw2sm_grav`, `Sm2gw_grav` arrays
- **Reason**: MODFLOW coupling not needed for standalone PRMS

### 3. Cascade Flow (Cascade_flag > CASCADE_OFF)
- **Fortran**: Lines ~716-743, ~1009-1037
- **Features skipped**:
  - `Upslope_interflow`, `Upslope_dunnianflow`
  - Cascading flow between HRUs
  - `compute_cascades()` subroutine
  - Lake inflows from soil zone
- **Reason**: May be added later if needed

### 4. Swale HRUs (compute_lateral==OFF)
- **Fortran**: Lines ~1050-1074
- **Features skipped**:
  - Special handling for swale HRUs without lateral flow
  - `Swale_actet` calculations
- **Implementation**: Basic swale logic retained (no lateral flow)

### 5. Basin Aggregations
- **Fortran**: All variables starting with `Basin_`
- **Examples**: `Basin_ag_soil_moist`, `Basin_ag_actet`, `Basin_soil_to_gw`
- **Reason**: Not needed; can be calculated post-processing if desired

### 6. MODSIM Integration
- **Fortran**: Lines ~589-598
- **Features**: `Model==MODSIM_PRMS` or `Model==MODSIM_PRMS_LOOSE`
- **Reason**: External model coupling not needed

### 7. Frozen Ground (CFGI)
- **Fortran**: Lines ~602-650, ~841-854
- **Features**: `Frozen_flag==ACTIVE`, `Frozen(i)==ACTIVE`
- **Features skipped**: Conditional frozen ground index logic
- **Reason**: Not needed for current applications

## Code Organization

### Class Structure

```python
class PRMSSoilzoneAg(ConservativeProcess):
    def __init__(...)              # Initialize with parameters and inputs
    def get_dimensions()           # Static: returns ("nhru",)
    def get_parameters()           # Static: lists all required parameters
    def get_inputs()               # Static: lists all required inputs
    def get_init_values()          # Static: initial values for all variables
    def get_restart_variables()   # Static: variables to save/restore
    def get_mass_budget_terms()   # Static: mass balance components
    
    def _initialize_soilzone_ag_data()  # Setup derived parameters
    def _init_calc_method()             # Setup calculation method
    def _advance_variables()            # Save previous timestep values
    def _calculate()                    # Main entry point with iteration loop
    
    @staticmethod
    def _calculate_numpy(...)           # Main HRU loop computation
    @staticmethod
    def _compute_soilmoist(...)         # Soil moisture accounting
    @staticmethod
    def _compute_interflow(...)         # Interflow calculation
    @staticmethod
    def _compute_gwflow(...)            # Groundwater flow
    @staticmethod
    def _compute_szactet(...)           # Actual ET calculation
```

### Calculation Flow

```
_calculate() [with iteration loop]
  ├── Initialize iteration variables
  ├── DO WHILE keep_iterating:
  │   ├── Restore initial conditions (if not first iteration)
  │   ├── Call _calculate_numpy()
  │   │   ├── FOR each HRU:
  │   │   │   ├── Handle LAKE HRUs separately
  │   │   │   ├── Compute preferential flow
  │   │   │   ├── Compute pervious soil moisture (_compute_soilmoist)
  │   │   │   ├── Compute ag soil moisture (_compute_soilmoist)
  │   │   │   ├── Compute slow interflow (_compute_interflow)
  │   │   │   ├── Compute groundwater flow (_compute_gwflow)
  │   │   │   ├── Compute ag ET (_compute_szactet with AET_external)
  │   │   │   ├── Compute pervious ET (_compute_szactet)
  │   │   │   ├── Check if more irrigation needed
  │   │   │   └── Update ag_irrigation_add if unsatisfied
  │   │   └── Calculate storage changes for mass budget
  │   ├── Check convergence
  │   └── Increment soil_iter
  └── Report convergence status
```

## Key Fortran-to-Python Mappings

| Fortran Variable | Python Variable | Notes |
|-----------------|-----------------|-------|
| `Soil_moist(i)` | `soil_moist[ihru]` | Pervious area soil moisture |
| `Ag_soil_moist(i)` | `ag_soil_moist[ihru]` | Agricultural area soil moisture |
| `Infil(i)` | `infil_hru[ihru]` | Infiltration to pervious area |
| `Infil_ag(i)` | `infil_ag[ihru]` | Infiltration to ag area |
| `AET_external(i)` | `AET_external[ihru]` | Observed AET |
| `Ag_irrigation_add(i)` | `ag_irrigation_add[ihru]` | Estimated irrigation |
| `Soil_iter` | `soil_iter` | Iteration counter |
| `keep_iterating` | `keep_iterating` | Boolean iteration flag |
| `Perv_frac` | `perv_frac` | Pervious fraction of HRU |
| `agfrac` | `agfrac` | Agricultural fraction of HRU |

## Mass Budget

The mass budget tracks:

**Inputs**:
- `infil_hru`: Infiltration to pervious area
- `infil_ag`: Infiltration to agricultural area

**Outputs**:
- `perv_actet_hru`: ET from pervious area (HRU basis)
- `hru_ag_actet`: ET from agricultural area (HRU basis)
- `perv_soil_to_gw`: Direct recharge from pervious soil
- `ag_soil_to_gw`: Direct recharge from ag soil
- `ssr_to_gw`: Flow from gravity reservoir to groundwater
- `slow_flow`: Interflow from slow (gravity) reservoir
- `dunnian_flow`: Excess flow to streams
- `pref_flow`: Preferential flow (fast interflow)

**Storage Changes**:
- `soil_rechr_change_hru`: Change in recharge zone (pervious)
- `soil_lower_change_hru`: Change in lower zone (pervious)
- `slow_stor_change`: Change in slow reservoir
- `pref_flow_stor_change`: Change in preferential flow storage

Note: Agricultural storage changes are implicitly included in the pervious changes via the area fraction multipliers.

## Testing Recommendations

1. **Basic Test**: Run without iteration (`iter_aet_flag=False`)
   - Should behave like base soilzone with dual areas
   - Compare pervious area outputs to base `PRMSSoilzone`

2. **Iteration Test**: Run with constant `AET_external`
   - Monitor `ag_irrigation_add` convergence
   - Check `soil_iter` count
   - Verify `ag_actet` matches `AET_external` within tolerance

3. **Mass Balance Test**: Verify budget closure
   - Inputs - Outputs - Storage Changes ≈ 0

4. **Parameter Sensitivity**: Test with different:
   - `max_soilzone_ag_iter` (5, 10, 20)
   - `soilzone_aet_converge` (1e-6, 1e-5, 1e-4)
   - `ag_soilwater_deficit_min` (0, 0.1, 0.5)

5. **Agricultural Fraction**: Test edge cases:
   - `ag_frac = 0` (no ag area)
   - `ag_frac = 1` (all ag, no pervious)
   - Mixed HRUs with varying `ag_frac`

## Known Limitations

1. **No Numba Version**: Only numpy calculation method implemented
   - Could be added later for performance if needed

2. **No Cascade Support**: Inter-HRU flow not implemented
   - May be needed for some watershed configurations

3. **Simplified Swales**: Basic implementation without full swale logic

4. **No GSFLOW Coupling**: Cannot interact with MODFLOW groundwater model

## Future Enhancements

If needed, the following could be added:

1. **Cascade Support**: Implement `compute_cascades()` logic
2. **GSFLOW Integration**: Add gravity reservoir interaction
3. **Numba Acceleration**: Create `_calculate_numba()` version
4. **Frozen Ground**: Implement CFGI logic
5. **Basin Aggregations**: Add basin-wide summary variables
6. **Dynamic Parameters**: Allow time-varying ag parameters

## Usage Example

```python
from pywatershed import PRMSSoilzoneAg, Control, Parameters

# Setup control and parameters
control = Control.load("control.yaml")
params = Parameters.load("parameters.nc")

# Create soilzone with iteration
soilzone_ag = PRMSSoilzoneAg(
    control=control,
    discretization=params,
    parameters=params,
    infil_hru=infil_hru,
    infil_ag=infil_ag,
    ag_frac=ag_frac,
    AET_external=aet_obs,
    iter_aet_flag=True,  # Enable AET matching
    max_soilzone_ag_iter=10,
    soilzone_aet_converge=1e-5,
    # ... other inputs
)

# Run for one timestep
soilzone_ag.advance()
soilzone_ag.calculate(simulation_time)

# Access results
irrigation = soilzone_ag.ag_irrigation_add
ag_et = soilzone_ag.ag_actet
iterations = soilzone_ag.soil_iter
```

## Contact

For questions or issues with this implementation, please refer to:
- Original Fortran: `gsflow_src/2.4.0/src/soilzone_ag.f90`
- Python implementation: `pywatershed/hydrology/prms_soilzone_ag.py`
- Base implementation: `pywatershed/hydrology/prms_soilzone.py`

---

**Implementation Date**: 2024
**Python Version**: 3.9+
**Dependencies**: numpy, pywatershed.base.ConservativeProcess
