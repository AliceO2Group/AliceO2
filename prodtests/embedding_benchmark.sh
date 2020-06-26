#!/bin/bash

# Benchmark for MC embedding. It is meant to be representative for targeted GRID usage.
# Using parallel execution wherever we can. Different scenarios for background production or
# caching + IO should be added here.

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
nevS=100   # nevents signal
nevBG=10   # nevents background
NCPUS=8    # default on GRID

# default interaction rates in kHz
intRate=50

generS="pythia8hf"
generBG="pythia8hi"

# default sim engine
engine="TGeant3"

# options to pass to every workflow
gloOpt=" -b --run "


Usage() 
{
  echo "Usage: ${0##*/} [-e TGeant3|TGeant4]"
  exit
}

while [ $# -gt 0 ] ; do
    case $1 in
	-e) engine=$2; shift 2 ;;
	-h) Usage ;;
	*) echo "Wrong input"; Usage;
    esac
done

echo "Running background simulation for $nevBG $collSyst events with $generBG generator and engine $engine"
taskwrapper sim_bg.log o2-sim -j ${NCPUS} -n"$nevBG" --configKeyValue "Diamond.width[2]=6." -g "$generBG" -e "$engine" -o o2simbg --skipModules ZDC
# extract time

echo "Running signal simulation for $nevS $collSyst events with $generS generator and engine $engine"
taskwrapper sim_s.log o2-sim -j ${NCPUS} -n"$nevS" --configKeyValue "Diamond.width[2]=6." -g "$generS" -e "$engine" -o o2sims --embedIntoFile o2simbg_Kine.root --skipModules ZDC
# extract time

echo "Running digitization for $intRate kHz interaction rate"
# We run the digitization in stages in order to use
# the given number of CPUs in a good way. The order and sequence here can be modified and improvements will be
# achievable when "REST" contains more parallelism. TPC and TRD should use the resources fully already.
intRate=$((1000*(intRate)));
taskwrapper digi_TPC.log o2-sim-digitizer-workflow --tpc-lanes ${NCPUS} --onlyDet TPC $gloOpt --sims o2simbg,o2sims -n ${nevS} --interactionRate $intRate --disable-mc
taskwrapper digi_TRD.log o2-sim-digitizer-workflow --onlyDet TRD $gloOpt --configKeyValues="TRDSimParams.digithreads=${NCPUS}" --sims o2simbg,o2sims -n ${nevS} --interactionRate $intRate --disable-mc
taskwrapper digi_REST.log o2-sim-digitizer-workflow --skipDet TRD,TPC $gloOpt --sims o2simbg,o2sims -n ${nevS} --interactionRate $intRate --disable-mc

# extract gain from embedding procedure and background reusage by analysing reported times:
transporttime_BG=`awk '/walltime/{print $2}' sim_bg.log_time`
transporttime_S=`awk '/walltime/{print $2}' sim_s.log_time`
digi_time_TPC=`awk '/walltime/{print $2}' digi_TPC.log_time`
digi_time_TRD=`awk '/walltime/{print $2}' digi_TRD.log_time`
digi_time_REST=`awk '/walltime/{print $2}' digi_REST.log_time`

echo "$transporttime_BG $transporttime_S $digi_time_TPC $digi_time_TRD $digi_time_REST" | awk -v nS="${nevS}" -v nB="${nevBG}" '//{noembed=($1*nS/(1.*nB) + $2) + $3 + $4 + $5;embed=($1 + $2) + $3 + $4 + $5; print "NBG "nB " NSignal "nS" Ordinary-time "noembed," Embed-time "embed" Ratio ",noembed/embed }'

# TODO: compare this to scenario where we fetch background events from the network!
# Run this actually containerized to fix the CPU + MEMORY space.

rm localhost*
