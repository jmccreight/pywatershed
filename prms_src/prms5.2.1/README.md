pywatershed compiles this PRMS 5.2.1 source on demand with
`pywatershed.utils.compile_prms()` (gfortran and gcc, `DBL_PREC=true`),
which installs the binary in the repository's `bin/` under the name
`pywatershed.utils.get_prms_exe_name()` expects. To rebuild, call it with
`force=True`. It requires `make`, `gfortran` and `gcc` on your `PATH`;
ifort is not supported (it cannot target arm64).

If that build fails, reproduce it by hand in this directory to see the
compiler output:

```shell
make clean
make FC=gfortran CC=gcc DBL_PREC=true
```

`DBL_PREC=true` is required by the pywatershed tests. The makefile writes
`bin/prms` here (`bin/prms.exe` on Windows); only `compile_prms()` installs
it where the tests look.
