#!/bin/bash

if [ "0$O2_ROOT" == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load O2/latest
fi

[[ -z $GEN_TOPO_MYDIR ]] && GEN_TOPO_MYDIR="$(dirname $(realpath $0))"
source $GEN_TOPO_MYDIR/setenv.sh || { echo "setenv.sh failed" 1>&2 && exit 1; }

ARGS_ALL="--session ${OVERRIDE_SESSION:-default} --shm-throw-bad-alloc 0 --no-cleanup"
if [[ $NUMAGPUIDS == 1 ]]; then
  TMPSIZE=$(expr $DDSHMSIZE \* 1024 \* 1024)
  ARGS_ALL+=" --shm-segment-id 2 --shm-segment-size $TMPSIZE --shm-mlock-segment 1 --shm-zero-segment 1"
else
  ARGS_ALL+=" --shm-segment-size $SHMSIZE"
fi
if [[ $NORATELOG == 1 ]]; then
  ARGS_ALL+=" --fairmq-rate-logging 0"
fi

o2-raw-file-reader-workflow $ARGS_ALL --loop $NTIMEFRAMES --delay $TFDELAY --input-conf $RAWINPUTDIR/rawAll.cfg --onlyDet $WORKFLOW_DETECTORS --configKeyValues "HBFUtils.nHBFPerTF=$NHBPERTF;" --max-tf 0  --raw-channel-config "name=dpl-chan,type=push,method=bind,address=ipc://${UDS_PREFIX}${INRAWCHANNAME},transport=shmem,rateLogging=0" $GLOBALDPLOPT --run
