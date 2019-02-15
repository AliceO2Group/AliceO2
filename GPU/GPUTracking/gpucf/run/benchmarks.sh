#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'
buildDir='build'

plotFiles=( \
    "clusterfinder.toml" \
    "cmpWithPacked.toml" \
    "cmpWithReset.toml" \
)

plotCmd='run/plot.py'

ssh $remoteTgt <<-ENDSSH
    cd $tgtDir
    make -sC$buildDir/release benchmark -j64
    ./$buildDir/release/bin/benchmark -scl -ddata/digits-big.txt -omeasurements
ENDSSH

scp -r $remoteTgt:$tgtDir/$measurementsDir .


for config in ${plotFiles[*]}
do
    $plotCmd $measurementsDir/$config
done
