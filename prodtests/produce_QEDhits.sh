#!/bin/bash

# Script to produce hits coming from QED background interactions.
# These hits will be overlayed on hits from normal events to produce
# combined digit output.

Usage()
{
  echo "Usage: ${0##*/} [-n Number of events ] [-e TGeant3|TGeant4]"
  exit
}

engine="TGeant3"
while [ $# -gt 0 ] ; do
    case $1 in
	-n) nev=$2;  shift 2 ;;
	-e) engine=$2; shift 2 ;;
	-h) Usage ;;
	*) echo "Wrong input"; Usage;
    esac
done

#---------------------------------------------------
echo "Running simulation for $nev $collSyst events with $gener generator and engine $engine"
#
# we only need passive material and QED sensitive detectors close to beampipe
#
set -x
o2-sim -j 8 -m PIPE ABSO DIPO SHIL ITS FT0 MFT -n"$nev" -e "$engine" --noemptyevents -g external --configKeyValues "GeneratorExternal.fileName=${O2_ROOT}/share/Generators/external/QEDLoader.C;SimCutParams.maxRTracking=300" -o o2simqed

root -q -b -l ${O2_ROOT}/share/macro/analyzeHits.C\(\"o2simqed\"\) > QEDhitstats.log
