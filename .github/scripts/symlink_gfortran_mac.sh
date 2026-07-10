#!/usr/bin/env bash

# Symlink gfortran runtime libraries from the active conda environment
# (gfortran is provided by environment.yml) into /usr/local/lib, which is
# on dyld's default fallback search path. This lets the prebuilt binaries
# in bin/ resolve @rpath/libgfortran.5.dylib etc. regardless of the rpaths
# they were built with.
#
# Requires the micromamba environment to be activated (CONDA_PREFIX set),
# so this must run after the setup-micromamba step.

if [ -z "${CONDA_PREFIX}" ]; then
    echo "CONDA_PREFIX is not set: activate the conda environment first"
    exit 1
fi

new_libdir="/usr/local/lib"
sudo mkdir -p "$new_libdir"
for lib in libgfortran.5.dylib libquadmath.0.dylib; do
    src="${CONDA_PREFIX}/lib/${lib}"
    if [ ! -f "$src" ]; then
        echo "Not found: $src"
        exit 1
    fi
    sudo ln -fs "$src" "${new_libdir}/${lib}"
done
