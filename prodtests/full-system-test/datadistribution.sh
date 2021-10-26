#!/bin/bash

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

if [[ `which StfBuilder 2> /dev/null | wc -l` == "0" ]]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load DataDistribution/latest
fi

# For benchmark only, do NOT copy&paste!
[[ $NUMAGPUIDS == 1 ]] && export DATADIST_SHM_DELAY=30

if [[ ! -z $DD_STARTUP_DELAY ]]; then
  sleep $DD_STARTUP_DELAY
fi

if [[ -z $INPUT_FILE_LIST ]]; then
  DD_INPUT_CMD="--data-source-dir ./raw/timeframe"
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

ARGS_ALL="--session default --severity $SEVERITY --shm-segment-id 2 --shm-segment-size 1000000 --no-cleanup"

eval StfBuilder --id stfb --transport shmem \
  --dpl-channel-name dpl-chan --channel-config "name=dpl-chan,type=push,method=bind,address=ipc://@$INRAWCHANNAME,transport=shmem,rateLogging=1" \
  $DD_INPUT_CMD \
  --data-source-rate=${TFRATE} \
  --data-source-regionsize=${DDSHMSIZE} \
  --data-source-headersize=2048 \
  --data-source-enable \
  --data-source-preread 5 \
  --shm-no-cleanup on \
  --shm-monitor false \
  --control=static \
  ${ARGS_ALL}
