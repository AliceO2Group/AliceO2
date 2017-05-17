#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: runRecoSim <cluster input file> <track output file>"
  exit 0
fi

# ===| define variables |=======================================================
if [ -z "$CA" ]; then
  CA=/data/Work/software/alicesw/HLTSA/HLT/TPCLib/tracking-ca/standalone/ca
fi

clusterInputfile=$1
outputTrackFile=$2

script=$(readlink -f $0)
macroDir=$(dirname $script)
clusterConversion=${macroDir}/convertClusters.C
trackConversion=${macroDir}/convertTracks.C
addInclude=${macroDir}/addInclude.C

# ===| create temporary output directory |======================================
outname=o2
outdir=events${outname}

test -d ${outname} || mkdir ${outdir}

# ===| convert clusters |=======================================================
cd $outdir
cmd="root.exe -b -q -l -n -x ${addInclude} ${clusterConversion}'+g(\"${clusterInputfile}\")'"
echo $cmd
eval $cmd

# ===| find tracks |============================================================
cd ..
cmd="${CA} -EVENTS ${outname} -CPU -WRITEBINARY -NOPROMPT"
echo $cmd
eval $cmd
mv output.bin $outdir

# ===| convert to tracks |======================================================
#cd $outdir
cmd="root.exe -b -q -l -n -x ${addInclude} ${trackConversion}'+g(\"${outdir}/output.bin\",\"${clusterInputfile}\", \"\", \"${outputTrackFile}\")'"
echo $cmd
eval $cmd

cd ..
