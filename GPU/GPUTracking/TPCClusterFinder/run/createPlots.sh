#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2

measurementsDir='measurements'

scp $remoteTgt:$tgtDir/$measurementsDir/in/* $measurementsDir/in


plotFiles=$(find $measurementsDir -name '*.toml')
plotCmd='run/plot.py'

for config in ${plotFiles[*]}
do
    $plotCmd $config
done
