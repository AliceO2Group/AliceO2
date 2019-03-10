#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'
buildDir='build'

plotFiles=( \
    "clusterfinder.toml" \
    "cmpWithPacked.toml" \
    "timeline.toml" \
)

plotCmd='run/plot.py'


ssh $remoteTgt <<-ENDSSH
    cd $tgtDir
    mkdir -p $measurementsDir/in
    mkdir -p $measurementsDir/out
    make -sC$buildDir/release benchmark -j64
    ./$buildDir/release/bin/benchmark -scl -ddata/digits-big.txt -o$measurementsDir/in
ENDSSH

scp $remoteTgt:$tgtDir/$measurementsDir/in/* $measurementsDir/in


for config in ${plotFiles[*]}
do
    $plotCmd $measurementsDir/$config
done
