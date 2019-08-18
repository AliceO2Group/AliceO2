#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'
buildDir='build'


ssh $remoteTgt <<-ENDSSH
    cd $tgtDir
    mkdir -p $measurementsDir/in
    mkdir -p $measurementsDir/out
    make -sC$buildDir/release benchmark -j64
    ./$buildDir/release/bin/benchmark -scl -ddata/digits_ev10_pythia8hi.bin -o$measurementsDir/in
ENDSSH

scp $remoteTgt:$tgtDir/$measurementsDir/in/* $measurementsDir/in


plotFiles=$(find $measurementsDir -name '*.toml')
plotCmd='run/plot.py'

for config in ${plotFiles[*]}
do
    $plotCmd $config
done
