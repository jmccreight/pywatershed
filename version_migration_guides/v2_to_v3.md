<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Migrating a project from pywatershed 2.x to 3.x](#migrating-a-project-from-pywatershed-2x-to-3x)
  - [Procedure](#procedure)
  - [Breaking changes](#breaking-changes)
    - [1. `budget_type` renamed `imbalance_behavior`](#1-budget_type-renamed-imbalance_behavior)
    - [2. Budget output filenames include the quantity](#2-budget-output-filenames-include-the-quantity)
    - [3. Variable `seg_width` renamed `seg_flow_width`](#3-variable-seg_width-renamed-seg_flow_width)
    - [4. `PRMSAtmosphere.calculate_transp_tindex` renamed `calc_transp_tindex`](#4-prmsatmospherecalculate_transp_tindex-renamed-calc_transp_tindex)
    - [5. `PRMSAtmosphere.get_variables()` returns more, silently](#5-prmsatmosphereget_variables-returns-more-silently)
    - [6. `FlowNodeMaker.get_node` takes `self`](#6-flownodemakerget_node-takes-self)
    - [7. `Control.load_prms` behavior changes, silently](#7-controlload_prms-behavior-changes-silently)
  - [Deprecated in 3.x, still working](#deprecated-in-3x-still-working)
  - [What did NOT change — do not go looking](#what-did-not-change--do-not-go-looking)
  - [Symptom to cause](#symptom-to-cause)
  - [New in 3.0.0 (not breaking)](#new-in-300-not-breaking)
  - [Provenance](#provenance)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Migrating a project from pywatershed 2.x to 3.x

This guide is self-contained instructions for updating code that uses
pywatershed 2.x to work with pywatershed 3.x. It is written to be
followed by a person, or given whole to an AI assistant with a request
like: "Read this guide and migrate my project."

The complete list of breaking changes is below — it was derived by
diffing the released versions (git tags `2.0.4` and `3.0.0`, source
code and test suite) and verifying each candidate by hand, not from
release notes alone. Trust it as exhaustive for the 2.x -> 3.0.0
boundary; changes after 3.0.0 are in the
[release notes](https://github.com/DOI-USGS/pywatershed/blob/main/doc/whats-new.rst)
under each 3.x version.

## Procedure

1. Confirm the situation: the project currently runs against
   pywatershed 2.x (any 2.x; the API did not change within 2.x) and
   the goal is 3.x. For other version hops this guide does not apply;
   see the release notes linked above.
2. Locate the project's code. Scan only files under version control
   (`git ls-files`, if the project is a git repository) — untracked
   scratch files produce false findings. Scan `.py` and `.ipynb`
   files, and any YAML/text configuration that sets pywatershed
   control options.
3. Run the scans in the table below. Report each hit with file, line,
   and the corresponding fix. Apply fixes only with the project
   owner's approval.
4. If the starting point is an error message rather than a planned
   upgrade, begin from the "Symptom to cause" table, then run the
   full scan anyway — more than one change may apply.
5. After edits: rerun the project's own tests or a known simulation.
   Check the two silent changes (items 5 and 7) by hand if the
   project touches those APIs — no scan fully catches them.

## Breaking changes

### 1. `budget_type` renamed `imbalance_behavior`

Scan: `budget_type` (also catches the `graph_budget_type` variant).

Everywhere the keyword argument `budget_type` existed — every process
class (`PRMSCanopy`, `PRMSSnow`, `PRMSRunoff`, `PRMSSoilzone`,
`PRMSGroundwater`, `PRMSChannel`, `Starfit`, their variants), and
`FlowGraph` — it is now `imbalance_behavior`. The same key in
`control.options` is renamed identically. The values (`"defer"`,
`None`, `"warn"`, `"error"`) are unchanged.

```python
# before
control.options["budget_type"] = "error"
proc = pws.PRMSGroundwater(control, dis, params, ..., budget_type="warn")
# after
control.options["imbalance_behavior"] = "error"
proc = pws.PRMSGroundwater(control, dis, params, ..., imbalance_behavior="warn")
```

The helper functions in `prms_channel_flow_graph` (e.g.
`prms_channel_flow_graph_to_model_dict`) have their own prefixed
keyword, renamed the same way:

```python
# before                          # after
graph_budget_type="warn",         graph_imbalance_behavior="warn",
```

### 2. Budget output filenames include the quantity

Scan: `_budget.nc`

`<ProcessName>_budget.nc` is now `<ProcessName>_mass_budget.nc`
(energy budgets, new in 3.0.0, write
`<ProcessName>_energy_budget.nc`). Any code opening budget output
files by name must change:

```python
# before
ds = xr.open_dataset(output_dir / "PRMSChannel_budget.nc")
# after
ds = xr.open_dataset(output_dir / "PRMSChannel_mass_budget.nc")
```

### 3. Variable `seg_width` renamed `seg_flow_width`

Scan: `seg_width`

The only variable removed from the metadata between 2.0.4 and 3.0.0,
and it was a rename. Affects output variable selection
(`netcdf_output_var_names`), reading output files, and any metadata
lookup by name.

### 4. `PRMSAtmosphere.calculate_transp_tindex` renamed `calc_transp_tindex`

Scan: `calculate_transp_tindex`

A method rename, nothing else changed.

### 5. `PRMSAtmosphere.get_variables()` returns more, silently

Scan: no reliable pattern — check by hand if the project calls
`get_variables()` on `PRMSAtmosphere` (or iterates
`.variables`) to select variables.

In 2.x the class declared its own variable list: a `tuple` of 14
names. In 3.x the override was removed and the base class derives the
list from `get_init_values()`: a `list` of 15 names — `tmax_sum` is
new. Code that loops over the result gets one more variable than
before and no error is raised.

### 6. `FlowNodeMaker.get_node` takes `self`

Scan: `def get_node`

The 2.x base class declared `get_node(control, index)` with no
`self` (a bug). Custom `FlowNodeMaker` subclasses that copied that
signature must add `self`:

```python
# before
def get_node(control, index):
# after
def get_node(self, control, index):
```

Subclasses that already took `self` (as all the pywatershed built-in
makers did) are unaffected.

### 7. `Control.load_prms` behavior changes, silently

Scan: `load_prms`

Two changes for code loading PRMS control files:

- The options `parameter_file`, `netcdf_output_dir`, and
  `streamflow_module` are unwrapped from their one-element lists only
  when exactly one value is present. With multiple values, the option
  is now a list where 2.x delivered only the first element.
- `keep_unused_options=True` no longer forces
  `warn_unused_options=True`; the two arguments act independently.

## Deprecated in 3.x, still working

Scan: `.budget` (attribute access on process or node objects)

The `budget` property on processes is a deprecated alias for the new
`mass_budget` (3.x adds `energy_budget` beside it). It works in 3.x
and is slated for removal at the next major release — update now:

```python
# before                                   # after
model.processes[pp].budget["_inputs"]      model.processes[pp].mass_budget["_inputs"]
```

## What did NOT change — do not go looking

Verified facts about the 2.0.4 -> 3.0.0 boundary; re-deriving them
wastes effort:

- No public class or function was removed or renamed in
  `pywatershed`'s top-level exports (17 names were added). Every 2.x
  `import`/`from pywatershed import ...` still works.
- No parameter was removed or renamed in the parameter metadata
  (23 were added).
- No variable changed units.
- `seg_width` (item 3) is the *only* variable removed from the
  variable metadata.
- `Control.__init__` has the same signature; positional callers are
  safe.

## Symptom to cause

| Error or symptom | Cause |
|---|---|
| `TypeError: ... unexpected keyword argument 'budget_type'` | item 1 |
| `NameError: 'budget_type' is not an available control option` | item 1 |
| `FileNotFoundError` on `*_budget.nc` | item 2 |
| `KeyError: 'seg_width'` or empty selection for it | item 3 |
| `AttributeError: ... no attribute 'calculate_transp_tindex'` | item 4 |
| Output contains an unrequested variable `tmax_sum` | item 5 |
| `TypeError` on `get_node` argument count | item 6 |
| A control option holds a list where a string is expected | item 7 |
| `DeprecationWarning` mentioning `budget` | deprecated alias |

## New in 3.0.0 (not breaking)

For adopting new capabilities after migrating, the released notebooks
are the worked examples (links pinned to the 3.0.0 release):

- [Multi-process models, stream temperature](https://github.com/DOI-USGS/pywatershed/blob/3.0.0/examples/01_multi-process_models.ipynb)
- [PRMS legacy models](https://github.com/DOI-USGS/pywatershed/blob/3.0.0/examples/02_prms_legacy_models.ipynb)
- [Restart capability](https://github.com/DOI-USGS/pywatershed/blob/3.0.0/examples/08_restart_streamflow.ipynb)
- [Flexible model output](https://github.com/DOI-USGS/pywatershed/blob/3.0.0/examples/09_model_output.ipynb)
- [Agricultural irrigation](https://github.com/DOI-USGS/pywatershed/blob/3.0.0/examples/10_ag_irrigation_use.ipynb)

The full feature list is the v3.0.0 section of the release notes.

## Provenance

Derived 2026-08-31 by diffing git tags `2.0.4` and `3.0.0`
(package source, static metadata, and the test suite, whose
migration in the same commits supplied the before/after examples),
with each candidate verified against the tagged sources by hand.
