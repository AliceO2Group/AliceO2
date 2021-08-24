#!/bin/bash

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

# This sets up the hardcoded configuration to run the full system workflow on the EPN
export SHMSIZE=$(( 112 << 30 ))
export NUMAGPUIDS=1
export OPTIMIZED_PARALLEL_ASYNC=1
export CTFINPUT=1
export SHMTHROW=0
export SEVERITY=error

if [ ! -f matbud.root ]; then
  echo matbud.root missing
  exit 1
fi

if [ "0$FST_TMUX_NOWAIT" != "01" ]; then
  ENDCMD="echo END; sleep 1000"
fi

if [ "0$FST_TMUX_LOGPREFIX" != "0" ]; then
  LOGCMD0=" &> ${FST_TMUX_LOGPREFIX}_0.log"
  LOGCMD1=" &> ${FST_TMUX_LOGPREFIX}_1.log"
fi

export TFDELAY=$(expr $TFDELAY \* 2)
export NTIMEFRAMES=$(expr $(expr $NTIMEFRAMES + 1) / 2)

rm -f /dev/shm/*fmq*

tmux \
    new-session  "sleep 0; NUMAID=0 $MYDIR/dpl-workflow.sh $LOGCMD0; $ENDCMD" \; \
    split-window "sleep 2; NUMAID=1 $MYDIR/dpl-workflow.sh $LOGCMD1; $ENDCMD" \; \
    select-layout even-vertical
