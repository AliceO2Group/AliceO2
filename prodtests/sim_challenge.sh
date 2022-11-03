#!/bin/bash

# A simple chain of algorithms from MC to reco (and analysis)

# ------------ LOAD UTILITY FUNCTIONS ----------------------------
. ${O2_ROOT}/share/scripts/jobutils.sh
# ----------- START WITH ACTUAL SCRIPT ---------------------------


if [ -z "$SHMSIZE" ]; then export SHMSIZE=10000000000; fi

# default run number
# (for now set to a pilot beam run until we have all CCDB objects for default unanchored MC)
runNumDef=300000

# default time stamp --> will be determined from run number during the sim stage
# startTimeDef=$(($(date +%s%N)/1000000))

# default number of events
nevPP=10
nevPbPb=10

# default interaction rates in kHz
intRatePP=400
intRatePbPb=50

# default collision system
collSyst="pp"

generPP="pythia8pp"
generPbPb="pythia8hi"

# default sim engine
engine="TGeant3"

# options to pass to every workflow
gloOpt=" -b --run --shm-segment-size $SHMSIZE"

# ITS reco options depends on pp or pbpb
ITSRecOpt=""

# option to set the number of sim workers
simWorker=""

# option to set the number of tpc-lanes
tpcLanes=""

Usage()
{
  echo "Usage: ${0##*/} [-s system /pp[Def] or pbpb/] [-r IR(kHz) /Def = $intRatePP(pp)/$intRatePbPb(pbpb)] [-n Number of events /Def = $nevPP(pp) or $nevPbPb(pbpb)/] [-e TGeant3|TGeant4] [-t startTime/Def = $startTimeDef] [-run runNumber/Def = $runNumDef] [-f fromstage sim|digi|reco /Def = sim]"
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
    -t) startTime=$2; shift 2 ;;
    -run) runNumber=$2; shift 2 ;;
    -h) Usage ;;
    *) echo "Wrong input"; Usage;
  esac
done

# convert to lower case (the bash construct ${collSyst,,} is less portable)
collSyst=`echo "$collSyst" | awk '{print tolower($0)}'`
if [ "$collSyst" == "pp" ]; then
    gener="$generPP"
    ITSRecOpt=" --configKeyValues \"ITSVertexerParam.phiCut=0.5;ITSVertexerParam.clusterContributorsCut=3;ITSVertexerParam.tanLambdaCut=0.2\""
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

[[ -z $startTime ]] && startTime=$startTimeDef
[[ -z $runNumber ]] && runNumber=$runNumDef

dosim="0"
dodigi="0"
dotrdtrap="0"
doreco="0"
# convert to lowercase
fromstage=`echo "$fromstage" | awk '{print tolower($0)}'`
if [ "$fromstage" == "sim" ]; then
  dosim="1"
  dodigi="1"
  dotrdtrap="1"
  doreco="1"
elif [ "$fromstage" == "digi" ]; then
  dodigi="1"
  dotrdtrap="1"
  doreco="1"
elif [ "$fromstage" == "reco" ]; then
  doreco="1"
else
  echo "Wrong stage string $fromstage provided, should be sim or digi or reco"
  Usage
fi


if [ "$dosim" == "1" ]; then
  #---- GRP creation ------
  echo "Creating GRPs ... and publishing in local CCDB overwrite"
  taskwrapper grp.log o2-grp-simgrp-tool createGRPs --run ${runNumber} --publishto GRP -o mcGRP

  #---------------------------------------------------
  echo "Running simulation for $nev $collSyst events with $gener generator and engine $engine and run number $runNumber"
  taskwrapper sim.log o2-sim -n"$nev" --configKeyValues "Diamond.width[2]=6." -g "$gener" -e "$engine" $simWorker --run ${runNumber}

  ##------ extract number of hits
  taskwrapper hitstats.log root -q -b -l ${O2_ROOT}/share/macro/analyzeHits.C
fi

if [ "$dodigi" == "1" ]; then
  echo "Running digitization for $intRate kHz interaction rate"
  intRate=$((1000*(intRate)));
  taskwrapper digi.log o2-sim-digitizer-workflow $gloOpt --interactionRate $intRate $tpcLanes --configKeyValues "HBFUtils.runNumber=${runNumber} --combine-devices"
  echo "Return status of digitization: $?"
  # existing checks
  #root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckDigits.C+
fi

if [ "$dotrdtrap" == "1" ]; then
  echo "Running TRD trap simulator"
  taskwrapper trdtrap.log o2-trd-trap-sim $gloOpt
  echo "Return status of trd trap sim: $?"
fi


if [ "$doreco" == "1" ]; then

  echo "Running TPC reco flow"
  #needs TPC digitized data
  taskwrapper tpcreco.log o2-tpc-reco-workflow $gloOpt --input-type digits --output-type clusters,tracks,send-clusters-per-sector  --configKeyValues "GPU_rec.maxTrackQPtB5=20"
  echo "Return status of tpcreco: $?"

  echo "Running ITS reco flow"
  taskwrapper itsreco.log  o2-its-reco-workflow --trackerCA --tracking-mode async $gloOpt $ITSRecOpt
  echo "Return status of itsreco: $?"

  # existing checks
  # root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckClusters.C+
  # root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckTracks.C+

  echo "Running MFT reco flow"
  #needs MFT digitized data
  MFTRecOpt=" --configKeyValues \"MFTTracking.forceZeroField=false;MFTTracking.LTFclsRCut=0.0100;\""
  taskwrapper mftreco.log  o2-mft-reco-workflow  $gloOpt $MFTRecOpt
  echo "Return status of mftreco: $?"

  echo "Running MCH reco flow"
  taskwrapper mchreco.log o2-mch-reco-workflow $gloOpt
  echo "Return status of mchreco: $?"

  echo "Running FT0 reco flow"
  #needs FT0 digitized data
  taskwrapper ft0reco.log o2-ft0-reco-workflow $gloOpt
  echo "Return status of ft0reco: $?"

  echo "Running FDD reco flow"
  #needs FDD digitized data
  taskwrapper fddreco.log o2-fdd-reco-workflow $gloOpt
  echo "Return status of fddreco: $?"

  echo "Running FV0 reco flow"
  #needs FV0 digitized data
  taskwrapper fv0reco.log o2-fv0-reco-workflow $gloOpt
  echo "Return status of fv0reco: $?"

  echo "Running MID reco flow"
  #needs MID digitized data
  taskwrapper midreco.log "o2-mid-digits-reader-workflow | o2-mid-reco-workflow $gloOpt"
  echo "Return status of midreco: $?"

  echo "Running MCH-MID matching flow"
  taskwrapper mchmidMatch.log "o2-muon-tracks-matcher-workflow $gloOpt"
  echo "Return status of mchmidmatch: $?"

  echo "Running ITS-TPC matching flow"
  #needs results of o2-tpc-reco-workflow, o2-its-reco-workflow and o2-fit-reco-workflow
  taskwrapper itstpcMatch.log o2-tpcits-match-workflow --use-ft0 $gloOpt
  echo "Return status of itstpcMatch: $?"

  echo "Running TRD matching to ITS-TPC and TPC"
  #needs results of o2-tpc-reco-workflow, o2-tpcits-match-workflow and o2-trd-tracklet-transformer
  taskwrapper trdTrkltTransf.log o2-trd-tracklet-transformer $gloOpt
  echo "Return status of trdTrkltTransf: $?"
  taskwrapper trdMatch.log o2-trd-global-tracking $gloOpt
  echo "Return status of trdTracker: $?"

  echo "Running MFT-MCH-MID matching flow"
  #needs results of o2-mch-reco-workflow, o2-mft-reco-workflow and o2-muon-tracks-matcher-workflow
  FwdMatchOpt=" --configKeyValues \"FwdMatching.useMIDMatch=true;\""
  taskwrapper mftmchMatch.log o2-globalfwd-matcher-workflow $gloOpt $FwdMatchOpt
  echo "Return status of globalfwdMatch: $?"

  echo "Running TOF reco flow to produce clusters"
  #needs results of TOF digitized data and results of o2-tpcits-match-workflow
  taskwrapper tofReco.log o2-tof-reco-workflow $gloOpt
  echo "Return status of tof cluster reco : $?"

  echo "Running Track-TOF macthing flow"
  #needs results of TOF clusters data from o2-tof-reco-workflow and results of o2-tpc-reco-workflow and ITS-TPC matching
  taskwrapper tofMatchTracks.log o2-tof-matcher-workflow $gloOpt
  echo "Return status of o2-tof-matcher-workflow: $?"

  echo "Running TOF matching QA"
  #need results of ITSTPC-TOF matching (+ TOF clusters and ITS-TPC tracks)
  taskwrapper tofmatch_qa.log root -b -q -l $O2_ROOT/share/macro/checkTOFMatching.C
  echo "Return status of TOF matching qa: $?"

  echo "Running ZDC reconstruction"
  #need ZDC digits
  taskwrapper zdcreco.log o2-zdc-digits-reco $gloOpt
  echo "Return status of ZDC reconstruction: $?"

  echo "Running EMC reconstruction"
  #need EMC digits
  taskwrapper emcreco.log o2-emcal-reco-workflow --infile emcaldigits.root $gloOpt
  echo "Return status of EMC reconstruction: $?"

  echo "Running PHS reconstruction"
  #need PHS digits
  taskwrapper phsreco.log o2-phos-reco-workflow $gloOpt
  echo "Return status of PHS reconstruction: $?"

  echo "Running CPV reconstruction"
  #need CPV digits
  taskwrapper cpvreco.log o2-cpv-reco-workflow $gloOpt
  echo "Return status of CPV reconstruction: $?"

  echo "Running primary vertex finding flow"
  #needs results of TPC-ITS matching and FIT workflows
  taskwrapper pvfinder.log o2-primary-vertexing-workflow $gloOpt --condition-remap file://./GRP=GLO/Config/GRPECS
  echo "Return status of primary vertexing: $?"

  echo "Running secondary vertex finding flow"
  #needs results of all trackers + P.Vertexer
  taskwrapper svfinder.log o2-secondary-vertexing-workflow $gloOpt
  echo "Return status of secondary vertexing: $?"

  echo "Producing AOD"
  taskwrapper aod.log o2-aod-producer-workflow $gloOpt --aod-writer-keep dangling --aod-writer-resfile "AO2D" --aod-writer-resmode UPDATE --aod-timeframe-id 1 --run-number 300000
  echo "Return status of AOD production: $?"

  # let's do some very basic analysis tests (mainly to enlarge coverage in full CI) and enabled when SIM_CHALLENGE_ANATESTING=ON
  if [[ ${O2DPG_ROOT} && ${SIM_CHALLENGE_ANATESTING} ]]; then
    # to be added again: Efficiency
    for t in ${ANATESTLIST:-MCHistograms Validation PIDTOF PIDTPC EventTrackQA WeakDecayTutorial}; do
      ${O2DPG_ROOT}/MC/analysis_testing/analysis_test.sh ${t}
      echo "Return status of ${t}: ${?}"
    done
  fi
fi
