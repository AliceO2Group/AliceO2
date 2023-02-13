#!/bin/bash

[[ -z $GEN_TOPO_MYDIR ]] && GEN_TOPO_MYDIR="$(dirname $(realpath $0))"
source $GEN_TOPO_MYDIR/setenv.sh

if [[ `which StfBuilder 2> /dev/null | wc -l` == "0" ]]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load DataDistribution/latest
fi

# For benchmark only, do NOT copy&paste!
[[ $NUMAGPUIDS == 1 ]] && [[ -z $SHM_MANAGER_SHMID ]] && export DATADIST_SHM_DELAY=10

if [[ ! -z $DD_STARTUP_DELAY ]]; then
  sleep $DD_STARTUP_DELAY
fi

if [[ -z $INPUT_FILE_LIST ]]; then
  DD_INPUT_CMD="--data-source-dir $RAWINPUTDIR/raw/timeframe"
else
  DD_INPUT_CMD="--data-source-file-list $INPUT_FILE_LIST"
  if [[ -z $INPUT_FILE_COPY_CMD ]]; then
    DD_INPUT_CMD+=" --data-source-copy-cmd \"XrdSecPROTOCOL=sss,unix xrdcp -N root://eosaliceo2.cern.ch/?src ?dst\""
  else
    DD_INPUT_CMD+=" --data-source-copy-cmd \"$INPUT_FILE_COPY_CMD\""
  fi
fi

if [[ $NTIMEFRAMES != -1 ]]; then
  export DATADIST_FILE_READ_COUNT=$NTIMEFRAMES
  DD_INPUT_CMD+=" --data-source-repeat"
fi
export TFRATE=$(awk "BEGIN {printf \"%.6f\",1/$TFDELAY}")

ARGS_ALL="--session ${OVERRIDE_SESSION:-default} --severity $SEVERITY --shm-segment-id 2 --shm-segment-size 1000000 --no-cleanup"

[[ ! -z $SHM_MANAGER_SHMID ]] && SHM_TOOL_OPTIONS=" --shmid $SHM_MANAGER_SHMID --data-source-region-shmid 100 --data-source-header-shmid 101"

eval StfBuilder --id stfb --discovery-partition FST --transport shmem \
  --dpl-channel-name dpl-chan --channel-config "name=dpl-chan,type=push,method=bind,address=ipc://${UDS_PREFIX}${INRAWCHANNAME},transport=shmem,rateLogging=1" \
  $DD_INPUT_CMD \
  --data-source-rate=${TFRATE} \
  --data-source-regionsize=${DDSHMSIZE} \
  --data-source-headersize=${DDHDRSIZE} \
  --data-source-enable \
  --data-source-preread 5 \
  --shm-no-cleanup on \
  --shm-monitor false \
  --control=static \
  ${ARGS_ALL} ${SHM_TOOL_OPTIONS}
