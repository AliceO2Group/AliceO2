#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'
resdir="$measurementsDir/in"
buildDir='build'

declare -a configs=(
    "--layoutPadMajor -o$resdir/padmajor_vega20.json"
    "--layoutTimeMajor -o$resdir/timemajor_vega20.json"
    "--layoutTiling4x4 -o$resdir/tiling4x4_vega20.json"
    "--layoutTiling4x4 --builderScratchPad -o$resdir/scratchpad4x4_vega20.json"
    "--layoutTimeMajor --builderScratchPad -o$resdir/scratchpadTime_vega20.json"
    "--layoutTimeMajor -o$resdir/timemajor_vega10.json -g1"
    "--layoutTiling4x4 --builderScratchPad -o$resdir/scratchpad4x4_vega10.json -g1"
    "--layoutTimeMajor -cpad -o$resdir/timemajor_vega20_pad.json"
    "--layoutTimeMajor -crandom -o$resdir/timemajor_vega20_rand.json"
    "--layoutTimeMajor -cfull -o$resdir/timemajor_vega20_full.json"
    "--layoutTiling4x4 -cpad --builderScratchPad -o$resdir/scratchpad4x4_vega20_pad.json"
    "--layoutTiling4x4 -crandom --builderScratchPad -o$resdir/scratchpad4x4_vega20_rand.json"
    "--layoutTiling4x4 -cfull --builderScratchPad -o$resdir/scratchpad4x4_vega20_full.json"
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
