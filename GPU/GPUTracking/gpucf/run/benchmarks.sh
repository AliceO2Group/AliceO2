#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'
resdir="$measurementsDir/in"
buildDir='build'

declare -a configs=(
    "--halfs --layoutTiling4x4 --builderScratchPad -o$resdir/scratchpad4x4.json"
    "--halfs --layoutTiling4x8 --builderScratchPad -o$resdir/scratchpad4x8.json"
    "--halfs --layoutTiling8x4 --builderScratchPad -o$resdir/scratchpad8x4.json"
    "--layoutTimeMajor -o$resdir/timemajor.json"
    "--layoutPadMajor -o$resdir/padmajor.json"
)

for cfg in "${configs[@]}"; do
    ssh $remoteTgt <<-ENDSSH
        cd $tgtDir
        mkdir -p $measurementsDir/in
        mkdir -p $measurementsDir/out
        make -sC$buildDir/release benchmark -j64
            ./$buildDir/release/bin/benchmark -scl -ddata/digits_ev10_pythia8hi.bin $cfg
ENDSSH
done

scp $remoteTgt:$tgtDir/$measurementsDir/in/* $measurementsDir/in


plotFiles=$(find $measurementsDir -name '*.toml')
plotCmd='run/plot.py'

for config in ${plotFiles[*]}
do
    $plotCmd $config
done
