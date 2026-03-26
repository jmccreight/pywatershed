<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [PRMS Humidity Bug Fix](#prms-humidity-bug-fix)
  - [Summary](#summary)
  - [Reasoning](#reasoning)
  - [Fix Details](#fix-details)
  - [Diff](#diff)
  - [Result](#result)
  - [pywatershed Changes](#pywatershed-changes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# PRMS Humidity Bug Fix

This document describes the fix for the PRMS 5.2.1.1 humidity bug documented in [humidity_bug_prms.md](humidity_bug_prms.md).

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

## pywatershed Changes

See the pywatershed section below for corresponding changes needed to use actual humidity data instead of reproducing the bug.

_TODO: Document pywatershed changes once implemented._

_TODO: Compare the impact of the change on the drb\\\_2yr simulations._
