#!/bin/bash

[[ -z $GEN_TOPO_MYDIR ]] && GEN_TOPO_MYDIR="$(dirname $(realpath $0))"

if [[ -z $1 || -z $2 ]]; then
  echo "ERROR: Command line arguments missing. Syntax: run-workflow-on-inputlist.sh [CTF | DD | TF] [name of file with list of files to be processed] [Timeout in seconds (optional: default = disabled)] [Log to stdout (optional: default = enabled)]"
  exit 1
fi

if [[ $1 == "DD" && `which StfBuilder 2> /dev/null | wc -l` -eq 0 ]] || [[ -z $O2_ROOT ]]; then
  echo "ERROR: DataDistribution or O2 environment not loaded"
  exit 1
fi

if [[ $2 != "LOCAL" && ! -f $2 ]]; then
  echo "ERROR: List file $2 not found"
  exit 1
fi

for i in EXTINPUT CTFINPUT RAWTFINPUT DIGITINPUT; do
  [[ -n ${!i} ]] && { echo "$i must not be set!"; exit 1; }
done

NUM_PROCS=0
PIDS_LOG=
RETVAL=0
START_TIME=`date +%s`
LOG_PREFIX="log_$(date +%Y%m%d-%H%M%S)_"

[[ $2 != "LOCAL" ]] && export INPUT_FILE_LIST=$2
[[ -z $OVERRIDE_SESSION ]] && export OVERRIDE_SESSION=default_$$_$RANDOM
[[ -z $INRAWCHANNAME ]] && export INRAWCHANNAME=tf-builder-$$-$RANDOM

SESSION_ID=`fairmq-shmmonitor --get-shmid --session $OVERRIDE_SESSION | cut -d':' -f2 | sed 's/^ *//'`
echo "SESSION_ID is $SESSION_ID"
rm -rf /dev/shm/fmq_$SESSION_ID*
if [[ "0$IGNORE_EXISTING_SHMFILES" != "01" && `ls /dev/shm/*fmq* 2> /dev/null | wc -l` -ne 0 ]]; then
  echo "ERROR: Existing SHM files (you can set IGNORE_EXISTING_SHMFILES=1 to ignore and allow multiple parallel reconstruction sessions)"
  exit 1
fi

cleanup_handler() {
  echo "Signal handler $1 received, propagating to childs"
  RETVAL=1
  for i in `seq 1 $NUM_PROCS`; do
    PID_VAR="PID$i"
    kill -s $1 ${!PID_VAR} 2> /dev/null
  done
}

start_process() {
  let NUM_PROCS=$NUM_PROCS+1
  $@ > ${LOG_PREFIX}$NUM_PROCS.log &
  PID=$!
  echo "Started $@ with PID $PID"
  eval PID$NUM_PROCS=$PID
}

trap "cleanup_handler INT" SIGINT
trap "cleanup_handler TERM" SIGTERM

echo "Processing $2 in $1 mode"

if [[ $1 == "DD" ]]; then
  export EXTINPUT=1
  export DD_STARTUP_DELAY=5
  start_process $GEN_TOPO_MYDIR/datadistribution.sh
elif [[ $1 == "CTF" ]]; then
  export CTFINPUT=1
elif [[ $1 == "TF" ]]; then
  export RAWTFINPUT=1
elif [[ $1 == "MC" ]]; then
  export DIGITINPUT=1
else
  echo "ERROR: Unsupported mode $1 requested"
  exit 1
fi

start_process ${DPL_WORKFLOW_FROM_OUTSIDE:-$GEN_TOPO_MYDIR/dpl-workflow.sh}

if [[ "0$4" != "00" ]]; then
  sleep 1
  # a sentinel that lives until all processes are done (or this script is gone); tail prints all logs and exits after it
  ( while kill -0 $$ 2> /dev/null; do sleep 0.5; done ) > /dev/null 2>&1 &
  PID_SENTINEL=$!
  tail -n 1000000 -f -s 0.1 --pid=$PID_SENTINEL ${LOG_PREFIX}*.log &
  PIDS_LOG="$PID_SENTINEL|$!"
  ln -sf ${LOG_PREFIX}${NUM_PROCS}.log latest.log
fi

TIMEOUT_PHASE=0
while [[ -n `jobs -rp | grep -v -x -E "${PIDS_LOG:-none}"` ]]; do
  sleep 1
  if [[ -n $3 && $(date +%s) -ge $(($START_TIME + $TIMEOUT_PHASE * 20 + $3)) ]]; then
    RETVAL=1
    let TIMEOUT_PHASE=$TIMEOUT_PHASE+1
    echo "Timeout reached ($3 seconds) - Sending SIGINT signal"
    cleanup_handler SIGINT
    killall -s SIGINT StfBuilder
    killall -s SIGINT o2-dpl-run
  fi
done

# end the sentinel and give tail up to 10 s to print the rest of the logs
if [[ "0$4" != "00" ]]; then
  kill $PID_SENTINEL 2> /dev/null
  for i in `seq 1 100`; do
    [[ -z `jobs -rp` ]] && break
    sleep 0.1
  done
  if [[ -n `jobs -rp` ]]; then
    echo "WARNING: stopping the log output of ${LOG_PREFIX}*.log, it may be incomplete" 1>&2
    kill `jobs -rp` 2> /dev/null
  fi
  wait ${PIDS_LOG//|/ } 2> /dev/null
fi

for i in `seq 1 $NUM_PROCS`; do
  [[ $RETVAL != 0 ]] && break
  PID_VAR="PID$i"
  wait ${!PID_VAR}
  RETVAL=$?
done

fairmq-shmmonitor --session $OVERRIDE_SESSION --cleanup

if [[ $RETVAL == 0 ]]; then
  echo "Done processing $2 in $1 mode"
else
  echo "Error processing $2 in $1 mode"
fi

exit $RETVAL
