#!/bin/bash

if [ "0$1" == "0" ]; then
  echo Please indicate whether to start with raw-reader [rr] or with DataDistribution [dd]
  exit 1
fi

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

# This sets up the hardcoded configuration to run the full system workflow on the EPN
export NGPUS=4
export GPUTYPE=HIP
export SHMSIZE=$(( 128 << 30 ))
export NUMAGPUIDS=1
export EXTINPUT=1
export EPNPIPELINES=1
export SYNCMODE=1

if [ $1 == "dd" ]; then
  export CMD=datadistribution.sh
else
  export CMD=raw-reader.sh
fi

rm -f /dev/shm/*fmq*

tmux \
    new-session "NUMAID=0 numactl --membind 0 --cpunodebind 0 $MYDIR/dpl-workflow.sh" \; \
    split-window "NUMAID=1 numactl --membind 1 --cpunodebind 1 $MYDIR/dpl-workflow.sh" \; \
    split-window "sleep 30; numactl --cpunodebind 0 --interleave=all $MYDIR/$CMD"
