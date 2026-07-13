<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [PRMS Humidity Bug Documentation](#prms-humidity-bug-documentation)
  - [Summary](#summary)
  - [Relevant Control Parameters](#relevant-control-parameters)
  - [A Red Herring: `humidity_cbh_flag` Default Value](#a-red-herring-humidity_cbh_flag-default-value)
  - [The Actual Bug: Flawed `ierr=2` Logic](#the-actual-bug-flawed-ierr2-logic)
    - [What Actually Happens](#what-actually-happens)
    - [The Intended Design](#the-intended-design)
  - [Test Case: drb_2yr](#test-case-drb_2yr)
  - [pywatershed Handling](#pywatershed-handling)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# PRMS Humidity Bug Documentation

## Summary

PRMS 5.2.1.1 contains a bug that prevents humidity data from being read when the stream temperature module is enabled with `strmtemp_humidity_flag=0` (indicating humidity should come from HRU-level CBH data). This results in stream temperature calculations using zero humidity values, which is physically unrealistic. The problem can easily be verified by looking at the output values of `Humidity_hru` and `Seg_humid` for PRMS, which are all zero.

## Relevant Control Parameters

There are several relevant control parameters in play that cause some confusion, so we (re)document them here. This is from the PRMS 5.2.1.1 release notes:

**`stream_temp_flag`**: Flag to activate simulation of stream temperature using the stream_temp module (0=off; 1=on).

**`strmtemp_humidity_flag`**: Flag to specify where humidity information is read for use by the stream_temp module:

- `0` = CBH File specified by control parameter `humidity_day`
- `1` = parameter `seg_humidity` (monthly values by segment)
- `2` = Data File with values assigned based on parameter `seg_humidity_sta`

**`humidity_cbh_flag`**: Flag to specify whether to read a CBH file with humidity values (0=no; 1=yes; default=0).

**`humidity_day`**: File name of the humidity CBH file; this can be a full or relative path.

To clarify: `humidity_cbh_flag` is a general-purpose infrastructure flag used by multiple modules (Penman-Monteith ET, Priestley-Taylor ET, and stream_temp) to control whether humidity data is loaded from a CBH file. `strmtemp_humidity_flag` is specific to stream_temp and specifies which of three sources to use. When `strmtemp_humidity_flag=0`, PRMS automatically sets `humidity_cbh_flag=ACTIVE` in order to load the CBH data.

## A Red Herring: `humidity_cbh_flag` Default Value

At first glance, the `humidity_cbh_flag` control parameter appears to be the culprit. It defaults to `0` (OFF), per both the documentation and as seen in `setup_cont.c`:

**File**: `prms_src/prms5.2.1.1/mmf/setup_cont.c`, lines 270-272

```c
lval = (long *)umalloc (sizeof (long));
lval[0] = 0;
decl_control_int_array ("humidity_cbh_flag", 1, lval);
```

**However, this is NOT the bug.** The default value is irrelevant because PRMS correctly overrides in the following section:

**File**: `prms_src/prms5.2.1.1/prms/call_modules.f90`, lines 715-721

```fortran
IF ( control_integer(Humidity_cbh_flag, 'humidity_cbh_flag')/=0 ) Humidity_cbh_flag = OFF
...
IF ( Et_flag==potet_pm_module .OR. Et_flag==potet_pt_module .OR. &
     (Stream_temp_flag==ACTIVE .AND. Strmtemp_humidity_flag==OFF) ) Humidity_cbh_flag = ACTIVE
```

When `Stream_temp_flag==ACTIVE` and `Strmtemp_humidity_flag==OFF` (confusing use of this `OFF` variable meaning "0", i.e., humidity should come from CBH), line 721 correctly sets `Humidity_cbh_flag = ACTIVE`. At the end of `call_modules.f90`, the flag is properly enabled. The bug occurs later.

## The Actual Bug: Flawed `ierr=2` Logic

The real bug is in `climate_hru.f90` where a signaling mechanism using `ierr=2` has flawed logic.

### What Actually Happens

**File**: `prms_src/prms5.2.1.1/prms/climate_hru.f90`, lines 493-508

```fortran
IF ( Humidity_cbh_flag==ACTIVE ) THEN
  IF ( control_string(Humidity_day, 'humidity_day')/=0 ) CALL read_error(5, 'humidity_day')
  ierr = 2 ! signals routine to ignore CBH file requirement and use a parameter
  CALL find_header_end(Humidity_unit, Humidity_day, 'humidity_day', ierr, 1, Cbh_binary_flag)
  IF ( ierr==1 ) THEN
    istop = 1
  ELSEIF ( ierr==2 ) THEN
    Humidity_cbh_flag = 0       ! <-- BUG: disables humidity after successful file open
  ELSE
    CALL find_current_time(Humidity_unit, Start_year, Start_month, Start_day, ierr, Cbh_binary_flag)
    ...
  ENDIF
ENDIF
```

The problem is when the file IS found. Here's the structure of `find_header_end()`:

**File**: `prms_src/prms5.2.1.1/prms/utils_prms.f90`, lines 96, 103, 113-180 (condensed)

```fortran
SUBROUTINE find_header_end(Iunit, Fname, Paramname, Iret, Cbh_flag, Cbh_binary_flag)
...
      INTEGER, INTENT(INOUT) :: Iret       ! ierr in caller becomes Iret here
...
IF ( Iret/=2 ) Iret = 0                    ! Iret==2 is preserved here
...
OPEN ( Iunit, FILE=trim(Fname), STATUS='OLD', IOSTAT=ios )
...
IF ( ios/=0 ) THEN
  ! file fails to open
  IF ( Iret==2 ) THEN
    Iret = 0                               ! file not found: Iret reset to 0
    ...
  ELSE
    Iret = 1                               ! error
  ENDIF
ELSE
  ! file opened successfully (ios==0)
  ! reads CBH file headers only, sets Iret=1 only if a read error occurs
  ! *** NO CODE sets Iret=0 on success ***
ENDIF
```

We see that `Iret` is the passed `ierr`, which was hard-coded to `2` before the call. The file opens successfully, so `ios==0`. Because `Iret` remains at `2`, we wind up in the ELSE block. The ELSE block only reads the CBH file headers and does not change `Iret` unless there's an issue reading the file. It also does not read the data — that would happen later in the RUN phase (lines 161-167) if `Humidity_cbh_flag` remained `ACTIVE`. So `find_header_end()` returns `Iret` to `ierr` unchanged, still `2`. Back in `climate_hru.f90` (in the previous code block), `ELSEIF (ierr==2)` triggers, setting `Humidity_cbh_flag = 0`.

**Successfully finding and opening the humidity file causes humidity reading to be disabled.**

This results in `Humidity_hru` and `Seg_humid` being zero for the run. Tracing how the zero values are arrived at: during the INIT phase, `Humidity_hru` is initialized to zero while `Humidity_cbh_flag` is still `ACTIVE`:

**File**: `prms_src/prms5.2.1.1/prms/climate_hru.f90`, line 398

```fortran
IF ( Humidity_cbh_flag==ACTIVE ) Humidity_hru = 0.0
```

Then later in INIT (line 500), the bug disables the flag:

```fortran
  ELSEIF ( ierr==2 ) THEN
    Humidity_cbh_flag = 0       ! <-- BUG: disables humidity after successful file open
```

During the RUN phase, the READ block never executes because the flag is now 0:

**File**: `prms_src/prms5.2.1.1/prms/climate_hru.f90`, lines 161-167

```fortran
IF ( Humidity_cbh_flag==ACTIVE ) THEN
  ...
  READ ( Humidity_unit, *, IOSTAT=ios ) yr, mo, dy, hr, mn, sec, (Humidity_hru(i), i=1,Nhru)
  ...
```

So `Humidity_hru` remains at 0.0 for the entire simulation and it is used for computing `seg_humid`:

**File**: `prms_src/prms5.2.1.1/prms/stream_temp.f90`, lines 784-787

```fortran
! Compute segment humidity if info is specified in CBH as time series by HRU
IF ( Strmtemp_humidity_flag==0 ) then
   Seg_humid(i) = Seg_humid(i) + Humidity_hru(j)*harea
endif
```

And then stream temperature calculations use `Seg_humid = 0.0` (zero relative humidity).

### The Intended Design

The intended design seems to be that by passing `ierr=2`, the humidity CBH file is marked as optional: if the file cannot be opened, fall back gracefully to parameter values instead of raising an error.

The comment in `find_header_end()` describes what happens when the CBH file is NOT found (`ios/=0`) and `Iret==2`:

**File**: `prms_src/prms5.2.1.1/prms/utils_prms.f90`, lines 121-126

```fortran
IF ( ios/=0 ) THEN
  IF ( Iret==2 ) THEN ! this signals climate_hru to ignore the Humidity CBH file, could add other files
    Iret = 0
    IF ( Print_debug>DEBUG_less ) &
         WRITE ( *, '(/,A,/,A,/,A)' ) 'WARNING, optional CBH file not found, will use associated parameter values'
```

## Test Case: drb_2yr

**Control file**: `pywatershed/data/drb_2yr/nhm_stream_temp.control`

Configuration:

- `stream_temp_flag = 1` (stream temperature module ON)
- `strmtemp_humidity_flag = 0` (humidity from CBH file)
- `humidity_day = ./rhavg.cbh` (file path specified and exists)
- `humidity_cbh_flag` (NOT in control file — irrelevant due to line 721 override)

Expected: Read humidity from `rhavg.cbh`.
Actual: Humidity data is NOT read, `Humidity_hru` and `seg_humid` remain 0.0.

## pywatershed Handling

The pywatershed implementation intentionally reproduces this bug for validation purposes.

**File**: `pywatershed/hydrology/prms_stream_temp.py`, lines 1022-1027

```python
# Hardcode to use HRU humidity (flag == 0) with zeros for now
humidity_hru_data = np.zeros(self.nhru)
seg_humidity_month = np.zeros(self.nsegment)
strmtemp_humidity_flag = 0
```

**File**: `autotest/test_prms_stream_temp.py`, lines 237-254

```python
if key == "humidity_hru":
    pass
    # The following is a solution that reproduces PRMS behavior where
    # multiple bugs result in humidity_hru being zero. The code after
    # it would hopefully be adopted once some clarity is had around the
    # bugs in PRMS.
    stream_temp_inputs[key] = adapter_factory(
        np.zeros(parameters.dimensions["nhru"], dtype=np.float64),
        variable_name=key,
        control=control,
    )
    # # humidity_hru comes from rhavg.nc in the simulation directory
    # # Use AdapterNetcdf directly to specify variable name in the file
    # stream_temp_inputs[key] = AdapterNetcdf(
    #     simulation["dir"] / "rhavg.nc",
    #     variable="rhavg",
    #     control=control,
    # )
```

The commented code shows the intended correct implementation once the PRMS bug is addressed.

See [humidity_bug_prms_fix.md](humidity_bug_prms_fix.md) for details on how the fixes were implemented.
