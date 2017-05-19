#!/bin/bash

if [ $# -lt 3 ]; then
  echo "usage: runTracking <fileInfo> <pedestalFile> <nevents>"
fi

fileInfo=$1
pedestalFile=$2
nevents=$3

clusterFile="$PWD/clusters.root"
trackFile="tracks.root"

script=$(readlink -f $0)
macroDir=$(dirname $script)
findClusters=${macroDir}/RawClusterFinder.C
runReco=${macroDir}/runRecoSim.sh
addInclude=${macroDir}/addInclude.C
cherenkovFile=$PWD/cherenkov.txt

# ===| find raw clusters |======================================================
cmd="root.exe -b -q -x -l  ${addInclude} $findClusters'+(\"$fileInfo\",\"$pedestalFile\",\"$clusterFile\",$nevents)'"
echo $cmd
eval $cmd

# ===| run reconstruction |=====================================================
#cmd="$runReco $clusterFile $trackFile $cherenkovFile"
#echo $cmd
#eval $cmd
