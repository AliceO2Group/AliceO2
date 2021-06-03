#!/bin/bash

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

if [ `which StfBuilder 2> /dev/null | wc -l` == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load DataDistribution/latest
fi

# For benchmark only, do NOT copy&paste!
export DATADIST_SHM_DELAY=10
export DATADIST_FILE_READ_COUNT=$NTIMEFRAMES

export TF_DIR=./raw/timeframe
export TFRATE=$(awk "BEGIN {printf \"%.6f\",1/$TFDELAY}")

ARGS_ALL="--session default --severity $SEVERITY --shm-segment-id 2 --shm-segment-size 1000000 --no-cleanup"

StfBuilder --id stfb --transport shmem \
  --dpl-channel-name dpl-chan --channel-config "name=dpl-chan,type=push,method=bind,address=ipc://@$INRAWCHANNAME,transport=shmem,rateLogging=1" \
  --data-source-dir ${TF_DIR} \
  --data-source-rate=${TFRATE} \
  --data-source-repeat \
  --data-source-regionsize=${DDSHMSIZE} \
  --data-source-headersize=1024 \
  --data-source-enable \
  --data-source-preread 5 \
  --shm-no-cleanup on \
  --shm-monitor false \
  --control=static \
  ${ARGS_ALL}
