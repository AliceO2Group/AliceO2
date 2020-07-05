#!/bin/bash

# chain of algorithms from MC and reco


# ----------- START WITH UTILITY FUNCTIONS ----------------------------

child_pid_list=
# finds out all the (recursive) child process starting from a parent
# output includes the parent
# output is saved in child_pid_list
childprocs() {
  local parent=$1
  if [ "$parent" ] ; then
    child_pid_list="$child_pid_list $parent"
    for childpid in $(pgrep -P ${parent}); do
      childprocs $childpid
    done;
  fi
}

# accumulate return codes
RC_ACUM=0

taskwrapper() {
  # A simple task wrapper launching a DPL workflow in the background 
  # and checking the output for exceptions. If exceptions are found,
  # all participating processes will be sent a termination signal.
  # The rational behind this function is to be able to determine failing 
  # conditions early and prevent longtime hanging executables 
  # (until DPL offers signal handling and automatic shutdown)

  local logfile=$1
  shift 1
  local command="$*"

  # launch the actual command in the background
  echo "Launching task: ${command} &> $logfile &"
  command="TIME=\"#walltime %e\" ${O2_ROOT}/share/scripts/monitor-mem.sh /usr/bin/time --output=${logfile}_time '${command}'"
  eval ${command} &> $logfile &

  # THE NEXT PART IS THE SUPERVISION PART
  # get the PID
  PID=$!

  while [ 1 ]; do
    # We don't like to see critical problems in the log file.

    # We need to grep on multitude of things:
    # - all sorts of exceptions (may need to fine-tune)  
    # - segmentation violation
    # - there was a crash
    pattern="-e \"xception\"                        \
             -e \"segmentation violation\"          \
             -e \"error while setting up workflow\" \
             -e \"There was a crash.\""
      
    grepcommand="grep -H ${pattern} $logfile >> encountered_exceptions_list 2>/dev/null"
    eval ${grepcommand}
    
    grepcommand="grep -h --count ${pattern} $logfile 2>/dev/null"
    # using eval here since otherwise the pattern is translated to a
    # a weirdly quoted stringlist
    RC=$(eval ${grepcommand})
    
    # if we see an exception we will bring down the DPL workflow
    # after having given it some chance to shut-down itself
    # basically --> send kill to all children
    if [ "$RC" != "" -a "$RC" != "0" ]; then
      echo "Detected critical problem in logfile $logfile"
      sleep 2

      # query processes still alive
      child_pid_list=
      childprocs ${PID}
      for p in $child_pid_list; do
        echo "killing child $p"
        kill $p
      done      

      RC_ACUM=$((RC_ACUM+1))
      return 1
    fi

    # check if command returned which may bring us out of the loop
    ps -p $PID > /dev/null
    [ $? == 1 ] && break

    # sleep for some time
    sleep 5
  done

  # wait for PID and fetch return code
  # ?? should directly exit here?
  wait $PID
  # return code
  RC=$?
  RC_ACUM=$((RC_ACUM+RC))
  [ ! "${RC} -eq 0" ] && echo "command ${command} had nonzero exit code ${RC}"

  return ${RC}
}

# ----------- START WITH ACTUAL SCRIPT ----------------------------


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
gloOpt=" -b --run "

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

collSyst="${collSyst,,}" # convert to lower case
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
fromstage="${fromstage,,}"
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
  root -q -b -l ${O2_ROOT}/share/macro/analyzeHits.C > hitstats.log
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
  taskwrapper tpcreco.log o2-tpc-reco-workflow $gloOpt --tpc-digit-reader \"--infile tpcdigits.root\" --input-type digits --output-type clusters,tracks  --tpc-track-writer \"--treename events --track-branch-name Tracks --trackmc-branch-name TracksMCTruth\"
  echo "Return status of tpcreco: $?"

  echo "Running ITS reco flow"
  taskwrapper itsreco.log  o2-its-reco-workflow  $gloOpt
  echo "Return status of itsreco: $?"

  # existing checks
  # root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckClusters.C+
  # root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckTracks.C+

  echo "Running MFT reco flow"
  #needs MFT digitized data
  taskwrapper mftreco.log  o2-mft-reco-workflow  $gloOpt
  echo "Return status of mftreco: $?"

  echo "Running FIT(FT0) reco flow"
  #needs FIT digitized data
  taskwrapper fitreco.log o2-fit-reco-workflow $gloOpt
  echo "Return status of fitreco: $?"

  echo "Running ITS-TPC macthing flow"
  #needs results of o2-tpc-reco-workflow, o2-its-reco-workflow and o2-fit-reco-workflow
  taskwrapper itstpcMatch.log o2-tpcits-match-workflow $gloOpt --tpc-track-reader \"tpctracks.root\" --tpc-native-cluster-reader \"--infile tpc-native-clusters.root\"
  echo "Return status of itstpcMatch: $?"

  echo "Running ITSTPC-TOF macthing flow"
  #needs results of TOF digitized data and results of o2-tpcits-match-workflow
  taskwrapper tofMatch.log o2-tof-reco-workflow $gloOpt
  echo "Return status of its-tpc-tof match: $?"

  echo "Running TOF matching QA"
  #need results of ITSTPC-TOF matching (+ TOF clusters and ITS-TPC tracks)
  root -b -q -l $O2_ROOT/share/macro/checkTOFMatching.C 1>tofmatch_qa.log 2>&1
  echo "Return status of TOF matching qa: $?"
fi

exit ${RC_ACUM}
