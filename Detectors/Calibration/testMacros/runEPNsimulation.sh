#!/bin/bash

iEPN=0
xpos_start=100
while [[ $iEPN -lt $1 ]]
do
    echo $iEPN
    xpos=$((xpos_start+1000*$iEPN))
    xterm -hold -geometry 150x41+$xpos+300 -e bash -c "echo $iEPN; o2-calibration-data-generator-workflow --lanes 7 --gen-slot $iEPN --gen-norm 3 --mean-latency 100000 --max-timeframes 500 -b | o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec downstream:TOF/CALIBDATA -b ; exec bash" &
    ((iEPN = iEPN +1 ))
done

