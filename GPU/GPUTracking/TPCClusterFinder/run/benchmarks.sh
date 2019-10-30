#!/bin/bash

set -euo pipefail

measurementsDir='measurements'
resdir="$measurementsDir/in"
buildDir='build'

declare -a configs=(
    "--layoutPadMajor -o$resdir/padmajor_vega20.json"
    "--layoutTimeMajor -o$resdir/timemajor_vega20.json"
    "--layoutTiling4x4 -o$resdir/tiling4x4_vega20.json"
    "--layoutTiling4x4 --builderScratchPad -o$resdir/scratchpad4x4_vega20.json"
    "--layoutTimeMajor --builderScratchPad -o$resdir/scratchpadTime_vega20.json"
    # "--layoutTimeMajor -o$resdir/timemajor_vega10.json -g1"
    # "--layoutTiling4x4 --builderScratchPad -o$resdir/scratchpad4x4_vega10.json -g1"
    "--layoutTimeMajor -cpad -o$resdir/timemajor_vega20_pad.json"
    "--layoutTimeMajor -crandom -o$resdir/timemajor_vega20_rand.json"
    "--layoutTimeMajor -cfull -o$resdir/timemajor_vega20_full.json"
    "--layoutTiling4x4 -cpad --builderScratchPad -o$resdir/scratchpad4x4_vega20_pad.json"
    "--layoutTiling4x4 -crandom --builderScratchPad -o$resdir/scratchpad4x4_vega20_rand.json"
    "--layoutTiling4x4 -cfull --builderScratchPad -o$resdir/scratchpad4x4_vega20_full.json"
    "--layoutTiling4x4 --builderScratchPad --wgSize=128 -o$resdir/scratchpad4x4_vega20_128.json"
    "--layoutTiling4x4 --builderScratchPad --wgSize=256 -o$resdir/scratchpad4x4_vega20_256.json"
)

mkdir -p $measurementsDir/in
mkdir -p $measurementsDir/out
make -sC$buildDir/release benchmark -j64

for cfg in "${configs[@]}"; do
    ./$buildDir/release/bin/benchmark -scl -ddata/digits_ev10_pythia8hi.bin $cfg
done
