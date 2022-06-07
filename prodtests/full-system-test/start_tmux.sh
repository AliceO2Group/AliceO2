#!/bin/bash

if [ "0$1" != "0dd" ] && [ "0$1" != "0rr" ] && [ "0$1" != "0tf" ]; then
  echo Please indicate whether to start with raw-reader [rr] or with DataDistribution [dd] or TfReader [tf]
  exit 1
fi

if [[ -z "${WORKFLOW_PARAMETERS+x}" ]]; then
  export WORKFLOW_PARAMETERS="CALIB,QC,EVENT_DISPLAY,CALIB_LOCAL_AGGREGATOR,CALIB_PROXIES"
  if [[ -z "${GEN_TOPO_WORKDIR}" ]]; then
    mkdir -p gen_topo_tmp
    export GEN_TOPO_WORKDIR=`pwd`/gen_topo_tmp
  fi
fi
[[ -z "${SEVERITY}" ]] && export SEVERITY="important"

MYDIR="$(dirname $(realpath $0))"
source $MYDIR/setenv.sh

# This sets up the hardcoded configuration to run the full system workflow on the EPN
export NGPUS=4
export GPUTYPE=HIP
export SHMSIZE=$(( 112 << 30 ))
export DDSHMSIZE=$(( 112 << 10 ))
export GPUMEMSIZE=$(( 24 << 30 ))
export NUMAGPUIDS=1
export EXTINPUT=1
export EPNPIPELINES=1
export SYNCMODE=1
export SHMTHROW=0
export IS_SIMULATED_DATA=1
export ALL_EXTRA_CONFIG="$ALL_EXTRA_CONFIG;NameConf.mCCDBServer=http://o2-ccdb.internal;"
export DPL_CONDITION_BACKEND="http://o2-ccdb.internal"
export DATADIST_NEW_DPL_CHAN=1

workflow_has_parameter QC && export QC_REDIRECT_MERGER_TO_LOCALHOST=1

if [ "0$FST_TMUX_MEM_OVERRIDE" != "0" ]; then
  export SHMSIZE=$(( $FST_TMUX_MEM_OVERRIDE << 30 ))
  export DDSHMSIZE=$(( $FST_TMUX_MEM_OVERRIDE << 10 ))
fi

if [ $1 == "dd" ]; then
  export CMD=datadistribution.sh
  export GPU_NUM_MEM_REG_CALLBACKS=5
elif [ $1 == "tf" ]; then
  export CMD=tf-reader.sh
  export GPU_NUM_MEM_REG_CALLBACKS=3
elif [ $1 == "rr" ]; then
  export CMD=raw-reader.sh
  export GPU_NUM_MEM_REG_CALLBACKS=3
fi

if [ ! -f matbud.root -a -f ctf_dictionary.root ]; then
  echo matbud.root or ctf_dictionary.root missing
  exit 1
fi

if [ "0$FST_TMUX_NOWAIT" != "01" ]; then
  ENDCMD="echo END; sleep 1000"
fi

if [ "0$FST_TMUX_KILLCHAINS" == "01" ]; then
  KILLCMD="sleep 60; ps aux | grep 'o2-dpl-run --session' | grep -v grep | awk '{print \$2}' | xargs kill -s INT --;"
fi

if [ "0$FST_TMUX_LOGPREFIX" != "0" ]; then
  LOGCMD0=" &> ${FST_TMUX_LOGPREFIX}_0.log"
  LOGCMD1=" &> ${FST_TMUX_LOGPREFIX}_1.log"
  LOGCMD2=" &> ${FST_TMUX_LOGPREFIX}_2.log"
  LOGCMD3=" &> ${FST_TMUX_LOGPREFIX}_3.log"
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
  CALIB_COMMAND="$MYDIR/aggregator-workflow.sh $LOGCMD3"
  CALIB_COMMAND_TMUX="split-window \"$CALIB_COMMAND; $ENDCMD\" ';'"
else
  CALIB_COMMAND=
  CALIB_COMMAND_TMUX=
fi

if [ "0$FST_TMUX_BATCH_MODE" == "01" ]; then
  { sleep $FST_SLEEP0; eval "NUMAID=0 $MYDIR/dpl-workflow.sh $LOGCMD0"; eval "$ENDCMD"; } &
  { sleep $FST_SLEEP1; eval "NUMAID=1 $MYDIR/dpl-workflow.sh $LOGCMD1"; eval "$ENDCMD"; } &
  { sleep $FST_SLEEP2; eval "SEVERITY=debug numactl --interleave=all $MYDIR/$CMD $LOGCMD2"; eval "$KILLCMD $ENDCMD"; } &
  { eval "$CALIB_COMMAND"; eval "$ENDCMD"; } &
  wait
else
  eval "tmux -L FST " \
    "new-session  \"sleep $FST_SLEEP0; NUMAID=0 $MYDIR/dpl-workflow.sh $LOGCMD0; $ENDCMD\" ';' " \
    "split-window \"sleep $FST_SLEEP1; NUMAID=1 $MYDIR/dpl-workflow.sh $LOGCMD1; $ENDCMD\" ';' " \
    ${CALIB_COMMAND_TMUX} \
    "split-window \"sleep $FST_SLEEP2; SEVERITY=debug numactl --interleave=all $MYDIR/$CMD; $KILLCMD $ENDCMD\" ';' " \
    "select-layout even-vertical"
fi

if [[ -z $SHM_MANAGER_SHMID ]]; then
  rm -f /dev/shm/*fmq*
fi
