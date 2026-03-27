<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [PRMS Humidity Bug Fix](#prms-humidity-bug-fix)
  - [Summary](#summary)
  - [Reasoning](#reasoning)
  - [Fix Details](#fix-details)
  - [Diff](#diff)
  - [Result](#result)
  - [Secondary Bug: `Seg_humid` Accumulation Reset Missing in v5.2.1.1](#secondary-bug-seg_humid-accumulation-reset-missing-in-v5211)
    - [Background: How `Seg_humid` is computed](#background-how-seg_humid-is-computed)
    - [The 5.2.1 design](#the-521-design)
    - [What changed in v5.2.1.1](#what-changed-in-v5211)
    - [Evidence this is an oversight](#evidence-this-is-an-oversight)
    - [Numeric effect](#numeric-effect)
    - [Proposed fix](#proposed-fix)
  - [pywatershed Changes](#pywatershed-changes)
    - [Background: `strmtemp_humidity_flag` Options](#background-strmtemp_humidity_flag-options)
    - [Class Structure](#class-structure)
    - [Design Rationale](#design-rationale)
    - [Implementation](#implementation)
    - [Test Cases](#test-cases)
  - [Next Steps](#next-steps)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# PRMS Humidity Bug Fix

This document describes the fix for the PRMS v5.2.1.1 humidity bug documented in [humidity_bug_prms.md](humidity_bug_prms.md).

## Summary

The fix removes the flawed `ierr=2` signaling mechanism that was causing humidity CBH file reading to be disabled even when the file was successfully found and opened.

## Reasoning

The original code used `ierr=2` as a signal to make the humidity CBH file "optional" — the intent seemed to be that if the file wasn't found, PRMS would fall back gracefully to parameter values. However:

1. **The mechanism was broken**: When the file WAS found, `ierr` remained at `2`, which triggered `Humidity_cbh_flag = 0`, disabling humidity reading.

2. **The mechanism was unnecessary**: The control file already requires `humidity_day` to be specified when `Humidity_cbh_flag==ACTIVE` (enforced by the `control_string()` call). A missing file should be an error, not silently ignored.

3. **Simpler is better**: Without `ierr=2`, the existing error handling works correctly:
   - File found → `ierr=0` → proceed with `find_current_time()` → success
   - File not found → `ierr=1` → `istop=1` → error (as it should be)

## Fix Details

**File**: `prms_src/prms5.2.1.1/prms/climate_hru.f90`

Two changes were made:

1. **Removed line 495**: `ierr = 2` — the signal that caused the bug
2. **Removed lines 499-500**: The `ELSEIF (ierr==2)` branch — now dead code

## Diff

```diff
         IF ( Humidity_cbh_flag==ACTIVE ) THEN
           IF ( control_string(Humidity_day, 'humidity_day')/=0 ) CALL read_error(5, 'humidity_day')
-          ierr = 2 ! signals routine to ignore CBH file requirement and use a parameter
           CALL find_header_end(Humidity_unit, Humidity_day, 'humidity_day', ierr, 1, Cbh_binary_flag)
           IF ( ierr==1 ) THEN
             istop = 1
-          ELSEIF ( ierr==2 ) THEN
-            Humidity_cbh_flag = 0
           ELSE
             CALL find_current_time(Humidity_unit, Start_year, Start_month, Start_day, ierr, Cbh_binary_flag)
             IF ( ierr==-1 ) THEN
               PRINT *, 'for first time step, CBH File: ', Humidity_day
               istop = 1
             ENDIF
           ENDIF
         ENDIF
```

## Result

After this fix:

- `Humidity_cbh_flag` remains `ACTIVE` when the humidity CBH file is found
- `Humidity_hru` is populated with actual data from the CBH file during the RUN phase
- `Seg_humid` receives real humidity values for stream temperature calculations
- If the humidity CBH file is missing, PRMS raises an error (correct behavior)

## Secondary Bug: `Seg_humid` Accumulation Reset Missing in v5.2.1.1

Fixing the CBH input bug (above) caused `Seg_humid` to receive real humidity values
for the first time. This in turn revealed a second, independent bug in
`stream_temp.f90`: the daily accumulation reset for `Seg_humid` under
`strmtemp_humidity_flag=0` (CBH mode) was silently dropped during the 5.2.1 →
5.2.1.1 refactor.

### Background: How `Seg_humid` is computed

`stream_temp.f90` supports three sources for segment humidity, selected by
`strmtemp_humidity_flag`:

| Flag | Source                                                  | How set each timestep                                                                        |
| ---- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `0`  | CBH file — daily per-HRU values (`humidity_hru`)        | Accumulated from HRUs in a loop, then divided by total HRU area to get area-weighted average |
| `1`  | Parameter — monthly per-segment values (`seg_humidity`) | Set directly from parameter array at start of timestep                                       |
| `2`  | Observation station data (`seg_humidity_sta`)           | Set directly from station array at start of timestep                                         |

For flags 1 and 2 the segment value is assigned outright each day — there is no
accumulation. For flag 0, `Seg_humid` must be zeroed before the HRU loop so
that area-weighted accumulation starts fresh every timestep, just like every other
accumulated variable (`Seg_ccov`, `Seg_melt`, `Seg_rain`, `Seg_tave_air`,
`hru_area_sum`).

### The 5.2.1 design

PRMS 5.2.1 handled this correctly with an `IF / ELSEIF / ELSE / ENDIF` block at
the top of the RUN section, immediately before all other variable resets:

```fortran
! Humidity info come from parameter file when Strmtemp_humidity_flag==1
! Otherwise it comes as daily values per HRU from CBH. Code for this is
! down in the HRU loop.
      IF ( Strmtemp_humidity_flag==1 ) THEN
         DO i = 1, Nsegment
            Seg_humid(i) = Seg_humidity(i, Nowmonth)
         ENDDO
      ELSEIF ( Strmtemp_humidity_flag==2 ) THEN ! use station data
         DO i = 1, Nsegment
            Seg_humid(i) = Humidity(Seg_humidity_sta(i))
         ENDDO
      ELSE
         Seg_humid = 0.0          ! <-- flag==0: zero before HRU accumulation
      ENDIF

      Seg_potet = 0.0D0
      Seg_ccov = 0.0
      Seg_melt = 0.0
      Seg_rain = 0.0
      hru_area_sum = 0.0
```

The `ELSE` branch is the reset for flag 0. The comment above the block explicitly
states the design intent: flags 1 and 2 are handled up front; flag 0 ("the CBH
case") is handled "down in the HRU loop" — meaning the loop only accumulates,
it does not initialise. The `ELSE` reset is what makes that work.

### What changed in v5.2.1.1

In v5.2.1.1 the block was restructured to handle the new `* 0.01` unit scaling for
flag 2. The `ELSE` branch was dropped, turning the block from
`IF / ELSEIF / ELSE / ENDIF` into `IF / ELSEIF / ENDIF`:

```fortran
! Humidity info come from parameter file when Strmtemp_humidity_flag==1
! Otherwise it comes as daily values per HRU from CBH. Code for this is
! down in the HRU loop.
      IF ( Strmtemp_humidity_flag==1 ) THEN
         DO i = 1, Nsegment
            Seg_humid(i) = Seg_humidity(i, Nowmonth)
         ENDDO
      ELSEIF ( Strmtemp_humidity_flag==2 ) THEN ! use station data
         DO i = 1, Nsegment
            Seg_humid(i) = Humidity(Seg_humidity_sta(i)) * 0.01
         ENDDO
      ENDIF                       ! <-- ELSE branch gone; no reset for flag==0

      Seg_potet = 0.0D0
      Seg_ccov = 0.0
      Seg_melt = 0.0
      Seg_rain = 0.0
      hru_area_sum = 0.0
```

The comment is **word-for-word identical** in both versions. The intent was not
changed — only the code was, and only because of the restructuring. The reset for
flag 0 was collateral damage.

### Evidence this is an oversight

1. **The comment was not updated.** It still says _"Code for this is down in the
   HRU loop"_ — i.e., flag 0 accumulates in the loop. An intentional carry-over
   design would have updated or removed that comment.

2. **Every other accumulated variable is reset.** `Seg_ccov`, `Seg_melt`,
   `Seg_rain`, `Seg_tave_air`, and `hru_area_sum` are all explicitly zeroed
   immediately after this block in both 5.2.1 and v5.2.1.1. `Seg_humid` under
   flag 0 follows exactly the same accumulation pattern and should be treated
   identically.

3. **The bug was latent.** Prior to the CBH input fix, `Humidity_hru` was always
   zero (due to the `ierr=2` bug in `climate_hru.f90`), so `Seg_humid` was zero
   every timestep regardless of whether it was reset. The missing reset only
   became observable once real CBH data started flowing in.

4. **The diff is a pure deletion.** The only change to this block in v5.2.1.1 was
   removing the `ELSE` branch. There is no added comment, no replacement logic,
   no indication that carry-over was a deliberate choice.

### Numeric effect

Without the reset, each timestep's computation for segments with HRUs is:

```
Seg_humid(i) = (Seg_humid_prev(i) + sum(Humidity_hru * 0.01 * area)) / hru_area_sum
```

where `Seg_humid_prev` is the previous timestep's finalised decimal-fraction value
(~0.6). The correct formula is:

```
Seg_humid(i) = sum(Humidity_hru * 0.01 * area) / hru_area_sum
```

The error per timestep is `Seg_humid_prev / hru_area_sum`. For large-area segments
(many or large HRUs) this is negligible; for small-area segments it can be
significant. In a 2-year DRB simulation (333,336 valid data points) the effect is:

- RMSE relative to the correct (reset) computation: **~0.030** decimal fraction
- R²: **0.918**

This is a systematic bias, not random noise, and it compounds over time.

### Proposed fix

In `prms_src/prms5.2.1.1/prms/stream_temp.f90`, reinstate the `ELSE` branch:

```diff
       IF ( Strmtemp_humidity_flag==1 ) THEN
          DO i = 1, Nsegment
             Seg_humid(i) = Seg_humidity(i, Nowmonth)
          ENDDO
       ELSEIF ( Strmtemp_humidity_flag==2 ) THEN ! use station data
          DO i = 1, Nsegment
             Seg_humid(i) = Humidity(Seg_humidity_sta(i)) * 0.01
          ENDDO
+      ELSE
+         Seg_humid = 0.0
       ENDIF
```

This restores the original 5.2.1 design, is consistent with all other accumulated
variables, and matches the stated intent of the comment above the block.

---

## pywatershed Changes

The pywatershed implementation requires a class restructuring to support different humidity sources. Because inputs are not optional in pywatershed, different humidity sources are handled by separate classes.

### Background: `strmtemp_humidity_flag` Options

PRMS supports three humidity sources for stream temperature:

- `0` = CBH File (time-varying `humidity_hru` input)
- `1` = Parameter `seg_humidity` (monthly values by segment)
- `2` = Data File with `seg_humidity_sta` (not implemented in pywatershed)

### Class Structure

**`PRMSStreamTempHumidityCBH`** (superclass, renamed from current `PRMSStreamTemp`):

- Corresponds to `strmtemp_humidity_flag=0`
- **Input**: `humidity_hru` — time-varying humidity data from CBH file
- Contains the full implementation details

**`PRMSStreamTemp`** (subclass, new):

- Corresponds to `strmtemp_humidity_flag=1`
- **Parameter**: `seg_humidity` — monthly humidity values by segment (12 values per segment)
- **No** `humidity_hru` input
- Calls into superclass methods with the parameter-based humidity values

### Design Rationale

1. **Inputs are not optional**: pywatershed class contracts require explicit inputs. A class cannot sometimes take an input and sometimes not.

2. **Superclass has full details**: The CBH case (`PRMSStreamTempHumidityCBH`) has the more complex, time-varying input and contains the full implementation.

3. **Subclass specializes**: `PRMSStreamTemp` inherits from `PRMSStreamTempHumidityCBH` and provides humidity via a static monthly parameter instead of a dynamic input.

4. **Simple name for common case**: The base name `PRMSStreamTemp` is used for the parameter-based case, which may be more common for users who don't have CBH humidity data.

### Implementation

Both classes are fully implemented in `pywatershed/hydrology/prms_stream_temp.py`.

**`PRMSStreamTempHumidityCBH`** (superclass):

- Renamed from the original `PRMSStreamTemp`
- `humidity_hru` is a required input, read from the CBH file (e.g. `rhavg.nc`)
- CBH humidity accumulation extracted into its own numba function
  `_compute_seg_humid_cbh_numba`, keeping the general aggregation function
  (`_compute_segment_aggregates_numba`) humidity-agnostic
- The accumulation reset (`seg_humid[:] = 0.0`) and the `* 0.01`
  percent-to-fraction scaling both match the corrected PRMS v5.2.1.1 Fortran

**`PRMSStreamTemp`** (subclass):

- Inherits from `PRMSStreamTempHumidityCBH`
- No `humidity_hru` input; instead uses `seg_humidity` parameter
- `seg_humidity` shape is `(nmonth, nsegment)`; scalar values are expanded
  automatically by the parameter loading machinery (`params_expand_scalar`)
- Overrides `_compute_segment_aggregates()` to call the shared numba function
  for all non-humidity variables, then sets `seg_humid` directly:
  `self.seg_humid[:] = self.seg_humidity[nowmonth - 1, :]`

### Test Cases

Three control files in `test_data/drb_2yr/` exercise all implemented paths.
All are handled by the single `test_compare_prms` function in
`autotest/test_prms_stream_temp.py`, which dispatches on `strmtemp_humidity_flag`:

| Control file                               | Flag | Class                       | `seg_humidity` source                                                                |
| ------------------------------------------ | ---- | --------------------------- | ------------------------------------------------------------------------------------ |
| `nhm_stream_temp.control`                  | `0`  | `PRMSStreamTempHumidityCBH` | Daily CBH (`rhavg.nc`)                                                               |
| `nhm_stream_temp_seg_humid_scalar.control` | `1`  | `PRMSStreamTemp`            | Scalar `0.62718494` (expands to all segs/months, this overall avg of the flag=0 run) |
| `nhm_stream_temp_seg_humid_matrix.control` | `1`  | `PRMSStreamTemp`            | Full `(nmonth, nsegment)` matrix computed as monthly averages of the flag=0 CBH run  |

The matrix parameter file (`myparam_seg_humid_matrix.param`) was generated by
`test_data/drb_2yr/make_seg_humid_matrix_param.py`, which groups the flag=0
`seg_humid.nc` output by calendar month and takes the mean.

## Next Steps

1. **PRMS-legacy instantiation**: Verify that notebook 02 and other
   PRMS-legacy workflows instantiate `PRMSStreamTempHumidityCBH` and
   `PRMSStreamTemp` correctly. Update any hardcoded `PRMSStreamTemp` references
   that should now point to one of the two classes depending on
   `strmtemp_humidity_flag`.

2. **Document output impacts on drb_2yr**: Compare `seg_tave_water` and related
   variables across the three test cases to quantify the effect of the humidity
   bug fixes. Key comparisons:
   - Flag=0 (corrected CBH) vs. the old zero-humidity baseline: shows the
     impact of actually reading humidity from the CBH file.
   - Flag=1 scalar vs. flag=1 matrix: the scalar `0.62718494` is the
     **overall mean** of daily `seg_humid` from the flag=0 run. The matrix
     case uses per-month averages and therefore captures the seasonal cycle of
     humidity, while the scalar collapses all variation into a single value.
   - Flag=1 matrix vs. flag=0 CBH: shows how well the smoothed monthly
     parameter approximates the full daily CBH signal in terms of stream
     temperature prediction.
