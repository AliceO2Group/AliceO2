#!/bin/bash

# Concatenate all parameters given on the command line to one 
# comma separated string 
parameters=""
for i in $*; do 
  if [ -z $parameters ]; then
    parameters=$i
  else
    parameters=$(echo "$parameters,$i")
  fi 
done

cd /home/ceres/klewin/O2/AliceO2/src/build/macro

if [ ! -e "./AliceO2_TGeant3.mc_$1_event.root" ]; then
  ./run_sim.sh $1
fi

if [ ! -e "./AliceO2_TGeant3.digi_$1_event.root" ]; then
  ./run_digi.sh $1
fi

if [ ! -e "./AliceO2_TGeant3.clusters_$1_event.root" ]; then
  ./run_clusterer.sh $1
  ./compare_cluster.sh $1
fi
