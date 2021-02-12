#!/bin/bash

MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

if [ "0$O2_ROOT" == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load O2/latest
fi

# Set general arguments
ARGS_ALL="--session default --severity $SEVERITY --shm-segment-id $NUMAID --shm-segment-size $SHMSIZE"
if [ $EXTINPUT == 1 ] || [ $NUMAGPUIDS == 1 ]; then
  ARGS_ALL+=" --no-cleanup"
fi
if [ $SHMTHROW == 0 ]; then
  ARGS_ALL+=" --shm-throw-bad-alloc 0"
fi
if [ $NORATELOG == 1 ]; then
  ARGS_ALL+=" --fairmq-rate-logging 0"
fi

# Set some individual workflow arguments depending on configuration
CTF_DETECTORS=ITS,MFT,TPC,TOF,FT0,MID,EMC,PHS,CPV
TPC_INPUT=zsraw
TPC_OUTPUT=tracks,clusters,disable-writer
TPC_CONFIG=
TPC_CONFIG_KEY=
TOF_INPUT=raw
TOF_OUTPUT=clusters,matching-info
ITS_CONFIG=
ITS_CONFIG_KEY=
if [ $SYNCMODE == 1 ]; then
  ITS_CONFIG_KEY+="fastMultConfig.cutMultClusLow=30;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500"
  TPC_CONFIG_KEY+=" GPU_global.synchronousProcessing=1;"
fi
if [ $CTFINPUT == 1 ]; then
  ITS_CONFIG+=" --tracking-mode async"
else
  ITS_CONFIG+=" --entropy-encoding"
  TOF_OUTPUT+=",ctf"
  TPC_OUTPUT+=",encoded-clusters"
fi

if [ $GPUTYPE == "HIP" ]; then
  if [ $NUMAID == 0 ] || [ $NUMAGPUIDS == 0 ]; then
    export TIMESLICEOFFSET=0
  else
    export TIMESLICEOFFSET=$NGPUS
  fi
  TPC_CONFIG_KEY+="GPU_proc.deviceNum=0;GPU_global.mutexMemReg=true;"
  TPC_CONFIG+=" --environment \"ROCR_VISIBLE_DEVICES={timeslice${TIMESLICEOFFSET}}\""
  export HSA_NO_SCRATCH_RECLAIM=1
  #export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
else
  TPC_CONFIG_KEY+="GPU_proc.deviceNum=-2;"
fi

if [ $GPUTYPE != "CPU" ]; then
  TPC_CONFIG_KEY+="GPU_proc.forceMemoryPoolSize=$GPUMEMSIZE;"
  if [ $HOSTMEMSIZE == "0" ]; then
    HOSTMEMSIZE=$(( 1 << 30 ))
  fi
fi
if [ $HOSTMEMSIZE != "0" ]; then
  TPC_CONFIG_KEY+="GPU_proc.forceHostMemoryPoolSize=$HOSTMEMSIZE;"
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

# Input workflow
if [ $CTFINPUT == 1 ]; then
  TPC_INPUT=compressed-clusters-ctf
  TOF_INPUT=digits
  WORKFLOW="o2-ctf-reader-workflow --ctf-input o2_ctf_0000000000.root --onlyDet $CTF_DETECTORS $ARGS_ALL | "
elif [ $EXTINPUT == 1 ]; then
  WORKFLOW="o2-dpl-raw-proxy $ARGS_ALL --dataspec \"B:TPC/RAWDATA;C:ITS/RAWDATA;D:TOF/RAWDATA;D:MFT/RAWDATA;E:FT0/RAWDATA;F:MID/RAWDATA;G:EMC/RAWDATA;H:PHS/RAWDATA;I:CPV/RAWDATA\" --channel-config \"name=readout-proxy,type=pull,method=connect,address=ipc://@stfb-to-dpl,transport=shmem,rateLogging=0\" | "
else
  WORKFLOW="o2-raw-file-reader-workflow $ARGS_ALL --configKeyValues \"HBFUtils.nHBFPerTF=$NHBPERTF;\" --delay $TFDELAY --loop $NTIMEFRAMES --max-tf 0 --input-conf rawAll.cfg | "
fi

#Decoder workflows
if [ $CTFINPUT == 0 ]; then
  WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --pipeline its-stf-decoder:$N_ITSDEC | "
  WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --runmft true | "
  WORKFLOW+="o2-ft0-flp-dpl-workflow $ARGS_ALL --disable-root-output | "
  WORKFLOW+="o2-mid-raw-to-digits-workflow $ARGS_ALL | "
  WORKFLOW+="o2-tof-compressor $ARGS_ALL | "
fi

# Common workflows
WORKFLOW+="o2-its-reco-workflow $ARGS_ALL --trackerCA $DISABLE_MC --clusters-from-upstream --disable-root-output $ITS_CONFIG --configKeyValues \"HBFUtils.nHBFPerTF=128;$ITS_CONFIG_KEY\" | "
WORKFLOW+="o2-tpc-reco-workflow ${ARGS_ALL/--severity $SEVERITY/--severity $SEVERITY_TPC} --input-type=$TPC_INPUT $DISABLE_MC --output-type $TPC_OUTPUT --pipeline tpc-tracker:$NGPUS,tpc-entropy-encoder:$N_TPCENT $TPC_CONFIG --configKeyValues \"HBFUtils.nHBFPerTF=$NHBPERTF;GPU_global.deviceType=$GPUTYPE;GPU_proc.debugLevel=0;$TPC_CONFIG_KEY\" | "
WORKFLOW+="o2-tpcits-match-workflow $ARGS_ALL --disable-root-input --disable-root-output $DISABLE_MC --pipeline itstpc-track-matcher:$N_TPCITS --configKeyValues \"HBFUtils.nHBFPerTF=128;\" | "
WORKFLOW+="o2-ft0-reco-workflow $ARGS_ALL --disable-root-input --disable-root-output $DISABLE_MC --configKeyValues \"HBFUtils.nHBFPerTF=128;\" | "
WORKFLOW+="o2-tof-reco-workflow $ARGS_ALL --configKeyValues \"HBFUtils.nHBFPerTF=$NHBPERTF\" --input-type $TOF_INPUT --output-type $TOF_OUTPUT --disable-root-input --disable-root-output $DISABLE_MC | "

# Workflows disabled in sync mode
if [ $SYNCMODE == 0 ]; then
  WORKFLOW+="o2-mid-reco-workflow $ARGS_ALL --disable-root-output $DISABLE_MC | "
  WORKFLOW+="o2-mft-reco-workflow $ARGS_ALL --clusters-from-upstream $DISABLE_MC --disable-root-output --configKeyValues \"HBFUtils.nHBFPerTF=128;\" | "
  WORKFLOW+="o2-primary-vertexing-workflow $ARGS_ALL $DISABLE_MC --disable-root-input --disable-root-output --validate-with-ft0 | "
  WORKFLOW+="o2-secondary-vertexing-workflow $ARGS_ALL --disable-root-input --disable-root-output | "
fi

# Workflows disabled in async mode
if [ $CTFINPUT == 0 ]; then
  WORKFLOW+="o2-phos-reco-workflow $ARGS_ALL --input-type raw --output-type cells $DISABLE_MC  | "
  WORKFLOW+="o2-cpv-reco-workflow $ARGS_ALL --input-type raw --output-type digits $DISABLE_MC  | "
  WORKFLOW+="o2-cpv-reco-workflow $ARGS_ALL --input-type digits --output-type clusters $DISABLE_MC | "
  WORKFLOW+="o2-emcal-reco-workflow $ARGS_ALL --input-type raw --output-type cells --disable-root-output $DISABLE_MC  | "

  WORKFLOW+="o2-itsmft-entropy-encoder-workflow $ARGS_ALL --runmft true | "
  WORKFLOW+="o2-ft0-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-mid-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-phos-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-cpv-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-emcal-entropy-encoder-workflow $ARGS_ALL | "

  WORKFLOW+="o2-tpc-scdcalib-interpolation-workflow $ARGS_ALL --disable-root-output --disable-root-input | "

  # Output workflow
  CTF_OUTPUT_TYPE="none"
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="both"; fi
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 0 ]; then CTF_OUTPUT_TYPE="dict"; fi
  if [ $CREATECTFDICT == 0 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="ctf"; fi
  CMD_CTF="o2-ctf-writer-workflow $ARGS_ALL --output-type $CTF_OUTPUT_TYPE --onlyDet $CTF_DETECTORS"
  if [ $CREATECTFDICT == 1 ] && [ $EXTINPUT == 1 ]; then
    CMD_CTF+=" --save-dict-after $NTIMEFRAMES"
  fi
  WORKFLOW+="$CMD_CTF | "
fi

# DPL run binary
WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT --run"

# Execute the command we have assembled
#echo Running workflow:
#echo $WORKFLOW | sed "s/| */|\n/g"
#echo
eval $WORKFLOW
