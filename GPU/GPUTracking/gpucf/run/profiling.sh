#!/bin/bash

set -euo pipefail

buildDir='build'

tmpDir='/tmp/gpucf_profiling'

rcpBin="$HOME/rcp/bin/rcprof"

rcpCommonFlags='-p -O -w .'


function profile()
{
    outfile=$1
    outfile=$tmpDir/$outfile
    cfconfig=$2

    sudo $rcpBin $rcpCommonFlags -o $outfile \
        $buildDir/release/bin/run_gpucf -scl -ddata/digits-big.txt \
        -o/tmp/clusters.txt $cfconfig
}


mkdir -p /tmp/gpucf_profiling

make -sC$buildDir/release run_gpucf -j64

profile Baseline.csv --std
profile TilingLayout.csv --tiling
profile IdxMacro.csv --idxMacro
profile PadMajor.csv --padMajor
