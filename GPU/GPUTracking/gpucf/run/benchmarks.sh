#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'
buildDir='build'

plotCmd='run/plot.py'

ssh $remoteTgt <<-ENDSSH
    cd $tgtDir
    make -sC$buildDir/release benchmark -j16
    ./$buildDir/release/bin/benchmark -scl -ddata/digits-big.txt -omeasurements
ENDSSH

scp -r $remoteTgt:$tgtDir/$measurementsDir .


$plotCmd -i=$measurementsDir/'naiveClusterFinder.csv' \
         -o=$measurementsDir/'naiveClusterFinder.pdf' \
         -y='time [ms]'
