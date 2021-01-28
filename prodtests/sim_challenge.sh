#!/bin/bash

# chain of algorithms from MC and reco


# ------------ LOAD UTILITY FUNCTIONS ----------------------------
. ${O2_ROOT}/share/scripts/jobutils.sh
# ----------- START WITH ACTUAL SCRIPT ---------------------------


# default number of events
nevPP=10
nevPbPb=10

# default interaction rates in kHz
intRatePP=400
intRatePbPb=50

# default collision system
collSyst="pp"

generPP="pythia8"
generPbPb="pythia8hi"

# default sim engine
engine="TGeant3"

# options to pass to every workflow
gloOpt=" -b --run --shm-segment-size 10000000000"

# option to set the number of sim workers
simWorker=""

# option to set the number of tpc-lanes
tpcLanes=""

Usage() 
{
  echo "Usage: ${0##*/} [-s system /pp[Def] or pbpb/] [-r IR(kHz) /Def = $intRatePP(pp)/$intRatePbPb(pbpb)] [-n Number of events /Def = $nevPP(pp) or $nevPbPb(pbpb)/] [-e TGeant3|TGeant4] [-f fromstage sim|digi|reco /Def = sim]"
  exit
}

fromstage="sim"
while [ $# -gt 0 ] ; do
    case $1 in
	-n) nev=$2;  shift 2 ;;
	-s) collSyst=$2; shift 2 ;;
	-r) intRate=$2; shift 2 ;;
	-e) engine=$2; shift 2 ;;
	-f) fromstage=$2; shift 2 ;;
        -j) simWorker="-j $2"; shift 2 ;;
        -l) tpcLanes="--tpc-lanes $2"; shift 2 ;;
	-h) Usage ;;
	*) echo "Wrong input"; Usage;
    esac
done

# convert to lower case (the bash construct ${collSyst,,} is less portable)
collSyst=`echo "$collSyst" | awk '{print tolower($0)}'`
if [ "$collSyst" == "pp" ]; then
    gener="$generPP"
    [[ "nev" -lt "1"  ]] && nev="$nevPP"
    [[ "intRate" -lt "1"  ]] && intRate="$intRatePP"
elif [ "$collSyst" == "pbpb" ]; then
    gener="$generPbPb"
    [[ "nev" -lt "1"  ]] && nev="$nevPbPb"
    [[ "intRate" -lt "1"  ]] && intRate="$intRatePbPb"
else
    echo "Wrong collision system $collSyst provided, should be pp or pbpb"
    Usage
fi

dosim="0"
dodigi="0"
doreco="0"
# convert to lowercase
fromstage=`echo "$fromstage" | awk '{print tolower($0)}'`
if [ "$fromstage" == "sim" ]; then
  dosim="1"
  dodigi="1"
  doreco="1"
elif [ "$fromstage" == "digi" ]; then
  dodigi="1"
  doreco="1"
elif [ "$fromstage" == "reco" ]; then
  doreco="1"
else
  echo "Wrong stage string $fromstage provided, should be sim or digi or reco"
  Usage
fi


if [ "$dosim" == "1" ]; then
  #---------------------------------------------------
  echo "Running simulation for $nev $collSyst events with $gener generator and engine $engine"
  taskwrapper sim.log o2-sim -n"$nev" --configKeyValue "Diamond.width[2]=6." -g "$gener" -e "$engine" $simWorker

  ##------ extract number of hits
  taskwrapper hitstats.log root -q -b -l ${O2_ROOT}/share/macro/analyzeHits.C
fi

if [ "$dodigi" == "1" ]; then
  echo "Running digitization for $intRate kHz interaction rate"
  intRate=$((1000*(intRate)));
  taskwrapper digi.log o2-sim-digitizer-workflow $gloOpt --interactionRate $intRate $tpcLanes
  echo "Return status of digitization: $?"
  # existing checks
  #root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckDigits.C+
fi

if [ "$doreco" == "1" ]; then
  echo "Running TPC reco flow"
  #needs TPC digitized data
  taskwrapper tpcreco.log o2-tpc-reco-workflow $gloOpt --input-type digits --output-type clusters,tracks
  echo "Return status of tpcreco: $?"

  echo "Running ITS reco flow"
  taskwrapper itsreco.log  o2-its-reco-workflow --trackerCA --tracking-mode async $gloOpt
  echo "Return status of itsreco: $?"

  # existing checks
  # root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckClusters.C+
  # root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckTracks.C+

  echo "Running MFT reco flow"
  #needs MFT digitized data
  taskwrapper mftreco.log  o2-mft-reco-workflow  $gloOpt
  echo "Return status of mftreco: $?"

  echo "Running FT0 reco flow"
  #needs FT0 digitized data
  taskwrapper ft0reco.log o2-ft0-reco-workflow $gloOpt
  echo "Return status of ft0reco: $?"

  echo "Running ITS-TPC macthing flow"
  #needs results of o2-tpc-reco-workflow, o2-its-reco-workflow and o2-fit-reco-workflow
  taskwrapper itstpcMatch.log o2-tpcits-match-workflow $gloOpt
  echo "Return status of itstpcMatch: $?"

  echo "Running ITSTPC-TOF macthing flow"
  #needs results of TOF digitized data and results of o2-tpcits-match-workflow
  taskwrapper tofMatch.log o2-tof-reco-workflow $gloOpt
  echo "Return status of its-tpc-tof match: $?"

  echo "Running primary vertex finding flow"
  #needs results of TPC-ITS matching and FIT workflows
  taskwrapper pvfinder.log o2-primary-vertexing-workflow $gloOpt
  echo "Return status of primary vertexing: $?"

  echo "Running TOF matching QA"
  #need results of ITSTPC-TOF matching (+ TOF clusters and ITS-TPC tracks)
  taskwrapper tofmatch_qa.log root -b -q -l $O2_ROOT/share/macro/checkTOFMatching.C
  echo "Return status of TOF matching qa: $?"

  echo "Producing AOD"
  taskwrapper aod.log o2-aod-producer-workflow --aod-writer-keep dangling --aod-writer-resfile "AO2D" --aod-writer-resmode UPDATE --aod-timeframe-id 1
  echo "Return status of AOD production: $?"
fi
