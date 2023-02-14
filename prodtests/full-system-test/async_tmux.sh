#!/bin/bash

[[ -z $GEN_TOPO_MYDIR ]] && GEN_TOPO_MYDIR="$(dirname $(realpath $0))"
source $GEN_TOPO_MYDIR/setenv.sh

# This sets up the hardcoded configuration to run the async full system test workflow on the EPN
if [ $GPUTYPE == "CPU" ]; then
  export SHMSIZE=$(( 112 << 30 ))
else
  export SHMSIZE=$(( 144 << 30 ))
fi
export NUMAGPUIDS=1
export OPTIMIZED_PARALLEL_ASYNC=1
export CTFINPUT=1
export SHMTHROW=0
export SEVERITY=error
export GPU_NUM_MEM_REG_CALLBACKS=2

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

export TFDELAY=$(($TFDELAY * 2))
export NTIMEFRAMES=$((($NTIMEFRAMES + 1) / 2))

rm -f /dev/shm/*fmq*

tmux -L ASYNC \
    new-session  "sleep 0; NUMAID=0 $GEN_TOPO_MYDIR/dpl-workflow.sh $LOGCMD0; $ENDCMD" \; \
    split-window "sleep 2; NUMAID=1 $GEN_TOPO_MYDIR/dpl-workflow.sh $LOGCMD1; $ENDCMD" \; \
    select-layout even-vertical
