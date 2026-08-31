<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Checked-in GSFLOW binaries](#checked-in-gsflow-binaries)
  - [Provenance](#provenance)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Checked-in GSFLOW binaries

GSFLOW source is not part of this repository, so these binaries are
checked in rather than compiled on demand (unlike PRMS, which is built
from `prms_src/`). All other binaries in this directory are gitignored.

## Provenance

All three GSFLOW 2.4.0 binaries were built from the same source commit
by the same CI run — gfortran (conda-forge `>=15.2.0,<16`), double
precision (`-freal-4-real-8`), gfortran runtime statically linked on
macOS and Linux:

- Repository: <https://github.com/jmccreight/gsflow_v2.4.0>
  (mirror of `code.usgs.gov/emorway/gsflow_v2.4.0`),
  branch `rebase_output_precision_pws`
- Commit: `7f61c53c027821a0d233ca07add9dc43e942557c`
- CI run: <https://github.com/jmccreight/gsflow_v2.4.0/actions/runs/33438130237>
  (artifacts `gsflow-<os>-latest-double`)

| File | Platform |
| --- | --- |
| `gsflow_2.4.0_gfortran_mac_arm_dbl_prec` | macOS arm64 (native, no Rosetta) |
| `gsflow_2.4.0_gfortran_linux_dbl_prec` | Linux x86_64 |
| `gsflow_2.4.0_gfortran_windows_dbl_prec.exe` | Windows x86_64 |

When replacing these binaries, take all platforms from a single CI run
and update this file.
