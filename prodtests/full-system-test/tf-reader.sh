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

TFName=`ls -t $RAWINPUTDIR/o2_*.tf 2> /dev/null | head -n1`
[[ -z $TFName && $WORKFLOWMODE == "print" ]] && TFName='$TFName'
[[ ! -z $INPUT_FILE_LIST ]] && TFName=$INPUT_FILE_LIST
if [[ -z $TFName && $WORKFLOWMODE != "print" ]]; then echo "No raw file given!"; exit 1; fi

o2-raw-tf-reader-workflow $ARGS_ALL --loop $NTIMEFRAMES --delay $TFDELAY --input-data ${TFName} ${INPUT_FILE_COPY_CMD+--copy-cmd} ${INPUT_FILE_COPY_CMD} --raw-channel-config "name=dpl-chan,type=push,method=bind,address=ipc://${UDS_PREFIX}${INRAWCHANNAME},transport=shmem,rateLogging=0" $GLOBALDPLOPT --run
