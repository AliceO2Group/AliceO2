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
export DDSHMSIZE=$(( 128 << 10 ))
export GPUMEMSIZE=$(( 24 << 30 ))
export NUMAGPUIDS=1
export EXTINPUT=1
export EPNPIPELINES=1
export SYNCMODE=1
export SHMTHROW=0
export SEVERITY=error

if [ $1 == "dd" ]; then
  export CMD=datadistribution.sh
else
  export CMD=raw-reader.sh
  export NTIMEFRAMES=1000000
fi

if [ ! -f matbud.root -a -f ctf_dictionary.root ]; then
  echo matbud.root or ctf_dictionary.root missing
  exit 1
fi

rm -f /dev/shm/*fmq*

tmux \
    new-session "NUMAID=0 $MYDIR/dpl-workflow.sh; echo END; sleep 1000" \; \
    split-window "sleep 30; NUMAID=1 $MYDIR/dpl-workflow.sh; echo END; sleep 1000" \; \
    split-window "sleep 60; SEVERITY=debug numactl --interleave=all $MYDIR/$CMD; echo END; sleep 1000" \; \
    select-layout even-vertical
