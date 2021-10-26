#!/bin/bash

MYDIR="$(dirname $(readlink -f $0))"

if [[ -z $1 || -z $2 ]]; then
  echo "ERROR: Command line arguments missing. Syntax: run-workflow-on-inputlist.sh [CTF | DD | TF] [name of file with list of files to be processed] [Timeout in seconds (optional: default = disabled)] [Log to stdout (optional: default = enabled)]"
  exit 1
fi

if [[ `which StfBuilder 2> /dev/null | wc -l` == "0" || -z $O2_ROOT ]]; then
  echo "ERROR: DataDistribution or O2 environment not loaded"
  exit 1
fi

if [[ $2 != "LOCAL" && ! -f $2 ]]; then
  echo "ERROR: List file $2 not found"
  exit 1
fi

NUM_PROCS=0
RETVAL=0
START_TIME=`date +%s`
LOG_PREFIX="log_$(date +%Y%m%d-%H%M%S)_"
if [ $2 != "LOCAL" ]; then
  export INPUT_FILE_LIST=$2
fi

rm -f ${LOG_PREFIX}*.log /dev/shm/*fmq*
if [[ `ls /dev/shm/*fmq* 2> /dev/null | wc -l` != "0" ]]; then
  echo "ERROR: Existing SHM files"
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
  export DD_STARTUP_DELAY=10
  start_process $MYDIR/datadistribution.sh
elif [[ $1 == "CTF" ]]; then
  export CTFINPUT=1
elif [[ $1 == "TF" ]]; then
  export RAWTFINPUT=1
else
  echo "ERROR: Unsupported mode $1 requested"
  exit 1
fi
start_process $MYDIR/dpl-workflow.sh

if [[ "0$4" != "00" ]]; then
  tail -f ${LOG_PREFIX}*.log &
  PID_LOG=$!
fi

TIMEOUT_PHASE=0
while [[ `jobs -rl | grep -v " $PID_LOG Running" | wc -l` != "0" ]]; do
  sleep 1
  if [[ ! -z $3 && $(date +%s) -ge $(($START_TIME + $TIMEOUT_PHASE * 20 + $3)) ]]; then
    RETVAL=1
    let TIMEOUT_PHASE=$TIMEOUT_PHASE+1
    echo "Timeout reached ($3 seconds) - Sending SIGINT signal"
    cleanup_handler SIGINT
    killall -s SIGINT StfBuilder
    killall -s SIGINT o2-dpl-run
  fi
done

if [[ "0$4" != "00" ]]; then
  kill $PID_LOG
fi

if [[ $RETVAL == 0 ]]; then
  echo "Done processing $2 in $1 mode"
else
  echo "Error processing $2 in $1 mode"
fi

exit $RETVAL
