#! /usr/bin/env bash

NRUNS=100
NJOBS=4
NEVENTS=1000
GEANT=TGeant3

shopt -s extglob

for I in $(seq -w 1 $NRUNS); do

    DIR="RUN$I"
    mkdir $DIR
    cp left_trace.macro $DIR/.
    cp primary_and_hits.macro $DIR/.
    cp secondary_and_hits.macro $DIR/.
    cd $DIR
    echo " --- starting run $I"
    o2-sim -j $NJOBS -n $NEVENTS -e $GEANT -g pythia8pp --skipModules ZDC --configKeyValues "Stack.pruneKine=false" &> o2-sim.log

    root -b -q -l "primary_and_hits.macro(\"o2sim_Kine.root\", \"barrel\")" &
    root -b -q -l "primary_and_hits.macro(\"o2sim_Kine.root\", \"muon\")" &
    root -b -q -l "primary_and_hits.macro(\"o2sim_Kine.root\", \"any\")" &

    root -b -q -l "secondary_and_hits.macro(\"o2sim_Kine.root\", \"barrel\")" &
    root -b -q -l "secondary_and_hits.macro(\"o2sim_Kine.root\", \"muon\")" &
    root -b -q -l "secondary_and_hits.macro(\"o2sim_Kine.root\", \"any\")" &

    wait

    rm !(*and_hits.*.root)

    cd ..
done
