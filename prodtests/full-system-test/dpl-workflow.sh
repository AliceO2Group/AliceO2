#!/bin/bash

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

if [ "0$ALIENVLVL" == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load O2/latest
fi

ARGS_ALL="--session default --severity $SEVERITY --shm-segment-id $NUMAID --shm-throw-bad-alloc 0 --shm-segment-size $SHMSIZE"

if [ $EXTINPUT == 1 ] || [ $NUMAGPUIDS == 1 ]; then
  ARGS_ALL+=" --no-cleanup"
fi

if [ $EXTINPUT == 1 ]; then
  CMD_X="B:TPC/RAWDATA;C:ITS/RAWDATA;D:TOF/RAWDATA;D:MFT/RAWDATA;E:FT0/RAWDATA;F:MID/RAWDATA"
  CMD_Y="name=readout-proxy,type=pull,method=connect,address=ipc://@stfb-to-dpl$1,transport=shmem,rateLogging=0"
  CMD_INPUT="o2-dpl-raw-proxy $ARGS_ALL --dataspec $CMD_X --channel-config $CMD_Y"
else
  CMD_X="HBFUtils.nHBFPerTF=$NHBPERTF;"
  CMD_INPUT="o2-raw-file-reader-workflow $ARGS_ALL --configKeyValues $CMD_X --delay $TFDELAY --loop $NTIMEFRAMES --max-tf 0 --input-conf rawAll.cfg"
fi

if [ $CREATECTFDICT == 1 ]; then
  CMD_DICT="o2-ctf-writer-workflow $ARGS_ALL --output-type dict --save-dict-after 1 --onlyDet ITS,MFT,TPC,TOF,FT0,MID"
else
  CMD_DICT=cat
fi

if [ $SYNCMODE == 1 ]; then
  CFG_X="fastMultConfig.cutMultClusLow=30;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500"
  ITS_CONFIG="--configKeyValues $CFG_X"
  TPC_CONFIG="GPU_global.synchronousProcessing=1;"
else
  ITS_CONFIG=
  TPC_CONFIG=
fi
TPC_CONFIG2=

if [ $GPUTYPE == "HIP" ]; then
  if [ $NUMAID == 0 ] || [ $NUMAGPUIDS == 0 ]; then
    export TIMESLICEOFFSET=0
  else
    export TIMESLICEOFFSET=$NGPUS
  fi
  TPC_CONFIG+="GPU_proc.deviceNum=0;GPU_global.mutexMemReg=true;"
  CFG_Y="ROCR_VISIBLE_DEVICES={timeslice${TIMESLICEOFFSET}}"
  TPC_CONFIG2+=" --environment $CFG_Y"
  export HSA_NO_SCRATCH_RECLAIM=1
  #export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
else
  TPC_CONFIG+="GPU_proc.deviceNum=-2;"
fi

if [ $GPUTYPE != "CPU" ]; then
  TPC_CONFIG+="GPU_proc.forceMemoryPoolSize=$GPUMEMSIZE;"
  if [ $HOSTMEMSIZE == "0" ]; then
    HOSTMEMSIZE=$(( 1 << 30 ))
  fi
fi
if [ $HOSTMEMSIZE != "0" ]; then
  TPC_CONFIG+="GPU_proc.forceHostMemoryPoolSize=$HOSTMEMSIZE;"
fi

if [ $EPNPIPELINES == 1 ]; then
  N_TPCENT=$(($(expr 7 \* $NGPUS / 4) > 0 ? $(expr 7 \* $NGPUS / 4) : 1))
  N_TPCITS=$(($(expr 7 \* $NGPUS / 4) > 0 ? $(expr 7 \* $NGPUS / 4) : 1))
  N_ITSDEC=$(($(expr 3 \* $NGPUS / 4) > 0 ? $(expr 3 \* $NGPUS / 4) : 1))
else
  N_TPCENT=1
  N_TPCITS=1
  N_ITSDEC=1
fi

$CMD_INPUT | \
o2-itsmft-stf-decoder-workflow $ARGS_ALL --pipeline its-stf-decoder:$N_ITSDEC | \
o2-itsmft-stf-decoder-workflow $ARGS_ALL --runmft true | \
o2-its-reco-workflow $ARGS_ALL --trackerCA $DISABLE_MC --clusters-from-upstream --disable-root-output --entropy-encoding $ITS_CONFIG | \
o2-itsmft-entropy-encoder-workflow $ARGS_ALL --runmft true | \
o2-tpc-reco-workflow ${ARGS_ALL/--severity $SEVERITY/--severity $SEVERITY_TPC} --input-type=zsraw $DISABLE_MC --output-type tracks,clusters,encoded-clusters,disable-writer --pipeline tpc-tracker:$NGPUS,tpc-entropy-encoder:$N_TPCENT $TPC_CONFIG2 --configKeyValues "HBFUtils.nHBFPerTF=$NHBPERTF;GPU_global.deviceType=$GPUTYPE;GPU_proc.debugLevel=0;$TPC_CONFIG" | \
o2-tpcits-match-workflow $ARGS_ALL --disable-root-input --disable-root-output $DISABLE_MC --pipeline itstpc-track-matcher:$N_TPCITS | \
o2-ft0-flp-dpl-workflow $ARGS_ALL --disable-root-output | \
o2-ft0-reco-workflow $ARGS_ALL --disable-root-input --disable-root-output $DISABLE_MC | \
o2-ft0-entropy-encoder-workflow $ARGS_ALL | \
o2-mid-raw-to-digits-workflow $ARGS_ALL | \
o2-mid-reco-workflow $ARGS_ALL --disable-root-output $DISABLE_MC | \
o2-mid-entropy-encoder-workflow $ARGS_ALL | \
o2-tof-compressor $ARGS_ALL | \
o2-tof-reco-workflow $ARGS_ALL --configKeyValues "HBFUtils.nHBFPerTF=$NHBPERTF" --input-type raw --output-type ctf,clusters,matching-info --disable-root-output $DISABLE_MC | \
o2-tpc-scdcalib-interpolation-workflow $ARGS_ALL --disable-root-output --disable-root-input | \
$CMD_DICT | \
o2-dpl-run $ARGS_ALL $GLOBALDPLOPT --run
