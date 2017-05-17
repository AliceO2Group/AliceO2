#!/bin/bash

if [ $# -lt 3 ]; then
  echo "usage: runTracking <fileInfo> <pedestalFile> <nevents>"
fi

fileInfo=$1
pedestalFile=$2
nevents=$3

clusterFile="clusters.root"
trackFile="tracks.root"

script=$(readlink -f $0)
macroDir=$(dirname $script)
findClusters=${macroDir}/RawClusterFinder.C
runReco=${macroDir}/runRecoSim.sh
addInclude=${macroDir}/addInclude.C

# ===| find raw clusters |======================================================
cmd="root.exe -b -q -x -l  ${addInclude} $findClusters'+(\"$fileInfo\",\"$pedestalFile\",\"$clusterFile\",$nevents)'"
echo $cmd
eval $cmd

# ===| run reconstruction |=====================================================
cmd="$runReco $clusterFile $trackFile"
echo $cmd
eval $cmd
