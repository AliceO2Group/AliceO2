#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'
buildDir='build'

plotCmd='run/plot.py'

ssh $remoteTgt <<-ENDSSH
    cd $tgtDir
    make -sC$buildDir/release benchmark -j64
    ./$buildDir/release/bin/benchmark -scl -ddata/digits-big.txt -omeasurements
ENDSSH

scp -r $remoteTgt:$tgtDir/$measurementsDir .


$plotCmd boxplot \
         -o=$measurementsDir/'clusterfinder.pdf' \
         -y='time [ms]' \
        $measurementsDir/'paddedClusterFinder.csv' \
        "Cluster Finder" \

$plotCmd boxplot \
         -o=$measurementsDir/'cmpPaddedAndPackedDigit.pdf' \
         -y='time [ms]' \
        $measurementsDir/'paddedClusterFinder.csv' \
        "With padded Digits" \
        $measurementsDir/'packedClusterFinder.csv' \
        "With packed Digits"

