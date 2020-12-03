#!/bin/bash

if [ "0$ALIENVLVL" == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load O2/latest
fi

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

ARGS_ALL="--session default --severity $SEVERITY --shm-throw-bad-alloc 0 --no-cleanup"
if [ $NUMAGPUIDS == 1 ]; then
  TMPSIZE=$(expr $DDSHMSIZE \* 1024 \* 1024)
  ARGS_ALL+=" --shm-segment-id 2 --shm-segment-size $TMPSIZE --shm-mlock-segment 1 --shm-zero-segment 1"
else
  ARGS_ALL+=" --shm-segment-size $SHMSIZE"
fi

o2-raw-file-reader-workflow $ARGS_ALL --loop $NTIMEFRAMES --delay $TFDELAY --input-conf rawAll.cfg --configKeyValues "HBFUtils.nHBFPerTF=$NHBPERTF;" --max-tf 0  --raw-channel-config "name=dpl-chan,type=push,method=bind,address=ipc://@stfb-to-dpl,transport=shmem,rateLogging=0" --run
