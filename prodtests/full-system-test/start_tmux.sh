#!/bin/bash

if [ "0$1" != "0dd" ] && [ "0$1" != "0rr" ] && [ "0$1" != "0tf" ]; then
  echo Please indicate whether to start with raw-reader [rr] or with DataDistribution [dd] or TfReader [tf] 1>&2
  exit 1
fi

if [[ -z "${WORKFLOW_PARAMETERS+x}" ]]; then
  export WORKFLOW_PARAMETERS="CALIB,QC,EVENT_DISPLAY,CALIB_LOCAL_AGGREGATOR"
  if [[ "0$FST_TMUX_INTEGRATED_AGGREGATOR" == "01" ]]; then
    export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS},CALIB_LOCAL_INTEGRATED_AGGREGATOR"
  else
    export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS},CALIB_PROXIES"
  fi
  if [[ -z "${GEN_TOPO_WORKDIR}" ]]; then
    mkdir -p gen_topo_tmp
    export GEN_TOPO_WORKDIR=`pwd`/gen_topo_tmp
  fi
fi
[[ -z "${SEVERITY}" ]] && export SEVERITY="important"

MYDIR="$(dirname $(realpath $0))"
source $MYDIR/setenv.sh

if [[ "0$FST_TMUX_NO_EPN" != "01" ]]; then
  # This sets up the hardcoded configuration to run the full system workflow on the EPN
  export NGPUS=4
  export GPUTYPE=HIP
  export SHMSIZE=$(( 112 << 30 ))
  export DDSHMSIZE=$(( 112 << 10 ))
  export GPUMEMSIZE=$(( 24 << 30 ))
  export NUMAGPUIDS=1
  export EPNPIPELINES=1
  export ALL_EXTRA_CONFIG="$ALL_EXTRA_CONFIG;NameConf.mCCDBServer=http://localhost:8084;"
  export DPL_CONDITION_BACKEND="http://localhost:8084"
  export GEN_TOPO_QC_OVERRIDE_CCDB_SERVER="http://localhost:8084"
  NUM_DPL_WORKFLOWS=2
  if [[ `lspci | grep "Vega 20" | wc -l` != "8" ]]; then
    echo "Could not detect 8 EPN GPUs, aborting" 1>&2
    exit 1
  fi
else
  [[ -z $NUM_DPL_WORKFLOWS ]] && NUM_DPL_WORKFLOWS=1
fi
export EXTINPUT=1
export SYNCMODE=1
export SHMTHROW=0
export IS_SIMULATED_DATA=1
export DATADIST_NEW_DPL_CHAN=1

workflow_has_parameter QC && export QC_REDIRECT_MERGER_TO_LOCALHOST=1

if [[ -z ${DPL_RAWPROXY_OVERRIDE_ORBITRESET+x} && $1 == "dd" ]]; then
  if [[ $BEAMTYPE == "PbPb" ]]; then
    export DPL_RAWPROXY_OVERRIDE_ORBITRESET=1550600800000
  else
    export DPL_RAWPROXY_OVERRIDE_ORBITRESET=1547590800000
  fi
fi

if [ "0$FST_TMUX_MEM_OVERRIDE" != "0" ]; then
  export SHMSIZE=$(( $FST_TMUX_MEM_OVERRIDE << 30 ))
  export DDSHMSIZE=$(( $FST_TMUX_MEM_OVERRIDE << 10 ))
fi

if [ $1 == "dd" ]; then
  export CMD=datadistribution.sh
  export GPU_NUM_MEM_REG_CALLBACKS=$(($NUM_DPL_WORKFLOWS + 3))
elif [ $1 == "tf" ]; then
  export CMD=tf-reader.sh
  export GPU_NUM_MEM_REG_CALLBACKS=$((NUM_DPL_WORKFLOWS + 1))
elif [ $1 == "rr" ]; then
  export CMD=raw-reader.sh
  export GPU_NUM_MEM_REG_CALLBACKS=$(($NUM_DPL_WORKFLOWS + 1))
fi

if [ "0$FST_TMUX_NOWAIT" != "01" ]; then
  ENDCMD="echo END; sleep 1000"
fi

if [ "0$FST_TMUX_KILLCHAINS" == "01" ]; then
  KILLCMD="sleep 60; ps aux | grep 'o2-dpl-run --session' | grep -v grep | awk '{print \$2}' | xargs kill -s INT --;"
fi

if [ "0$FST_TMUX_LOGPREFIX" != "0" ]; then
  LOGCMD=" &> ${FST_TMUX_LOGPREFIX}_[REPLACE].log"
fi

FST_SLEEP0=0
FST_SLEEP1=0
FST_SLEEP2=45
if [[ -z $SHM_MANAGER_SHMID ]]; then
  rm -f /dev/shm/*fmq*
  if [[ `ls /dev/shm/*fmq* 2> /dev/null | wc -l` != "0" ]]; then
    echo "FMQ SHM files left which cannot be deleted, please clean up!"
    exit 1
  fi
else
  FST_SLEEP0=0
  FST_SLEEP1=0
  FST_SLEEP2=30
fi

if [[ ! -z $FST_TMUX_SINGLENUMA ]]; then
  eval "FST_SLEEP$((FST_TMUX_SINGLENUMA ^ 1))=\"0; echo SKIPPED; sleep 1000; exit\""
  export GPU_NUM_MEM_REG_CALLBACKS=$(($GPU_NUM_MEM_REG_CALLBACKS - 1))
fi

if workflow_has_parameter CALIB_PROXIES; then
  CALIB_COMMAND="$MYDIR/aggregator-workflow.sh"
  CALIB_TASKS="BARREL_TF BARREL_SPORADIC CALO_TF" # CALO_SPORADIC MUON_TF MUON_SPORADIC
else
  CALIB_TASKS=
fi

if [ "0$FST_TMUX_BATCH_MODE" == "01" ]; then
  { sleep $FST_SLEEP0; eval "NUMAID=0 $MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/0}"; eval "$ENDCMD"; } &
  { sleep $FST_SLEEP1; eval "NUMAID=1 $MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/1}"; eval "$ENDCMD"; } &
  { sleep $FST_SLEEP2; eval "SEVERITY=debug numactl --interleave=all $MYDIR/$CMD ${LOGCMD/\[REPLACE]/2}"; eval "$KILLCMD $ENDCMD"; } &
  for i in $CALIB_TASKS; do
    { eval "AGGREGATOR_TASKS=$i $CALIB_COMMAND ${LOGCMD/\[REPLACE]/3_${i}}"; eval "$ENDCMD"; } &
  done
  wait
else
  TMUX_SPLIT_COMMAND="split-window"
  TMUX_COMMAND="tmux -L FST"
  TMUX_COMMAND+=" new-session  \"sleep $FST_SLEEP0; NUMAID=0 $MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/0}; $ENDCMD\" ';'"
  for i in `seq 1 $(($NUM_DPL_WORKFLOWS - 1))`; do
    TMUX_COMMAND+=" $TMUX_SPLIT_COMMAND \"sleep $FST_SLEEP1; NUMAID=$i $MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/1}; $ENDCMD\" ';'"
  done
  TMUX_COMMAND+=" $TMUX_SPLIT_COMMAND \"sleep $FST_SLEEP2; SEVERITY=debug numactl --interleave=all $MYDIR/$CMD; $KILLCMD $ENDCMD\" ';'"
  FIRST_CALIB=1
  for i in $CALIB_TASKS; do
    TMUX_COMMAND+=" $TMUX_SPLIT_COMMAND \"AGGREGATOR_TASKS=$i $CALIB_COMMAND ${LOGCMD/\[REPLACE]/3_${i}}; $ENDCMD\" ';'"
    if [[ $FIRST_CALIB == 1 ]]; then
      TMUX_COMMAND+=" select-layout even-vertical ';'"
      TMUX_SPLIT_COMMAND="split-window -h"
      FIRST_CALIB=0
    fi
  done
  [[ $FIRST_CALIB == 1 ]] && TMUX_COMMAND+=" select-layout even-vertical ';'"
  # echo "Running $TMUX_COMMAND"
  eval $TMUX_COMMAND
fi

if [[ -z $SHM_MANAGER_SHMID ]]; then
  rm -f /dev/shm/*fmq*
fi
