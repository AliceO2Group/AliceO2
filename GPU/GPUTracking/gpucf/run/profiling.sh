#!/bin/bash

set -euo pipefail

buildDir='build'

tmpDir='/tmp/gpucf_profiling'

rcpBin="$HOME/rcp/bin/rcprof"

rcpCommonFlags='-p -O -w .'


function profile()
{
    outfile=$1
    shift
    outfile=$tmpDir/$outfile
    cfconfig=$@

    sudo $rcpBin $rcpCommonFlags -o $outfile \
        $buildDir/release/bin/run_gpucf -scl -ddata/digits-big.txt \
        -o/tmp/clusters.txt $cfconfig
}


mkdir -p /tmp/gpucf_profiling

make -sC$buildDir/release run_gpucf -j64

profile Baseline.csv
profile TilingLayout.csv --tiling4x4
profile PadMajor.csv --padMajor

profile Halfs.csv --halfs
profile HalfsPadMajor.csv --halfs --padMajor
profile Halfs4x8Tiling.csv --halfs --tiling4x8
profile Halfs8x4Tiling.csv --halfs --tiling8x4
