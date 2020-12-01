#!/bin/bash

if [ "0$ALIENVLVL" == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load O2/latest
fi

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

ARGS_ALL="--session default --severity $SEVERITY --shm-throw-bad-alloc 0 --shm-segment-size $SHMSIZE --no-cleanup"

o2-raw-file-reader-workflow $ARGS_ALL --loop $NTIMEFRAMES --delay $TFDELAY --input-conf rawAll.cfg --configKeyValues "HBFUtils.nHBFPerTF=$NHBPERTF;" --max-tf 0  --raw-channel-config "name=dpl-chan,type=push,method=bind,address=ipc://@stfb-to-dpl,transport=shmem,rateLogging=0"
