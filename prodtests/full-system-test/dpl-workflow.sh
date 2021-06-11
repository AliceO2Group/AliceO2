#!/bin/bash

# In order to use o2sim_grp.root, o2sim_geometry.root or matbud.root from arbitrary directory one can provide to the workflow
# --configKeyValues "NameConf.mDirGRP=<dirname1>;NameConf.mDirGeom=<dirname2>;NameConf.mDirMatLUT=<dirname3>;"
# All workflows currently running in the FST parce the configKeyValues option, so the NameConf.mDirXXX keys can be added via global env.var.

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
if [ $NUMAGPUIDS != 0 ]; then
  ARGS_ALL+=" --child-driver 'numactl --membind $NUMAID --cpunodebind $NUMAID'"
fi

# Set some individual workflow arguments depending on configuration
CTF_DETECTORS=ITS,MFT,TPC,TOF,FT0,MID,EMC,PHS,CPV,ZDC,FDD,HMP,FV0,TRD
CTF_DIR=
CTF_DICT_DIR=
GPU_INPUT=zsraw
GPU_OUTPUT=tracks,clusters
GPU_CONFIG=
GPU_CONFIG_KEY=
TOF_INPUT=raw
TOF_OUTPUT=clusters,matching-info
ITS_CONFIG=
ITS_CONFIG_KEY=
TRD_CONFIG=
if [ $SYNCMODE == 1 ]; then
  ITS_CONFIG_KEY+="fastMultConfig.cutMultClusLow=30;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500;"
  GPU_CONFIG_KEY+="GPU_global.synchronousProcessing=1;GPU_proc.clearO2OutputFromGPU=1;"
  TRD_CONFIG+=" --tracking-sources ITS-TPC"
else
  TRD_CONFIG+=" --tracking-sources TPC,ITS-TPC"
fi
if [ $CTFINPUT == 1 ]; then
  ITS_CONFIG+=" --tracking-mode async"
else
  ITS_CONFIG+=" --entropy-encoding"
  TOF_OUTPUT+=",ctf"
  GPU_OUTPUT+=",compressed-clusters-ctf"
fi

if [ $GPUTYPE == "HIP" ]; then
  if [ $NUMAID == 0 ] || [ $NUMAGPUIDS == 0 ]; then
    export TIMESLICEOFFSET=0
  else
    export TIMESLICEOFFSET=$NGPUS
  fi
  GPU_CONFIG_KEY+="GPU_proc.deviceNum=0;GPU_global.mutexMemReg=true;"
  GPU_CONFIG+=" --environment \"ROCR_VISIBLE_DEVICES={timeslice${TIMESLICEOFFSET}}\""
  export HSA_NO_SCRATCH_RECLAIM=1
  #export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
else
  GPU_CONFIG_KEY+="GPU_proc.deviceNum=-2;"
fi

if [ $GPUTYPE != "CPU" ]; then
  GPU_CONFIG_KEY+="GPU_proc.forceMemoryPoolSize=$GPUMEMSIZE;"
  if [ $HOSTMEMSIZE == "0" ]; then
    HOSTMEMSIZE=$(( 1 << 30 ))
  fi
fi
if [ $HOSTMEMSIZE != "0" ]; then
  GPU_CONFIG_KEY+="GPU_proc.forceHostMemoryPoolSize=$HOSTMEMSIZE;"
fi

if [ $EPNPIPELINES != 0 ]; then
  N_TPCENT=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_TPCITS=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_ITSDEC=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_EMC=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_CPV=$(($(expr 5 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 5 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
else
  N_TPCENT=1
  N_TPCITS=1
  N_ITSDEC=1
  N_EMC=1
  N_CPV=1
fi

# Input workflow
if [ $CTFINPUT == 1 ]; then
  GPU_INPUT=compressed-clusters-ctf
  TOF_INPUT=digits
  CTFName=`ls -t o2_ctf_*.root | head -n1`
  CTF_DICT=
  if [ ! -z $CTF_DICT_DIR ] ; then CTF_DICT=" --ctf-dict ${CTF_DICT_DIR}/ctf_dictionary.root"; fi
  WORKFLOW="o2-ctf-reader-workflow --ctf-input ${CTFName}  ${CTF_DICT} --onlyDet $CTF_DETECTORS $ARGS_ALL  | "
elif [ $EXTINPUT == 1 ]; then
  WORKFLOW="o2-dpl-raw-proxy $ARGS_ALL --dataspec \"FLP:FLP/DISTSUBTIMEFRAME/0;B:TPC/RAWDATA;C:ITS/RAWDATA;D:TOF/RAWDATA;D:MFT/RAWDATA;E:FT0/RAWDATA;F:MID/RAWDATA;G:EMC/RAWDATA;H:PHS/RAWDATA;I:CPV/RAWDATA;J:ZDC/RAWDATA;K:HMP/RAWDATA;L:FDD/RAWDATA;M:TRD/RAWDATA;N:FV0/RAWDATA\" --channel-config \"name=readout-proxy,type=pull,method=connect,address=ipc://@$INRAWCHANNAME,transport=shmem,rateLogging=0\" | "
else
  WORKFLOW="o2-raw-file-reader-workflow --detect-tf0 $ARGS_ALL --configKeyValues \"HBFUtils.nHBFPerTF=$NHBPERTF;\" --delay $TFDELAY --loop $NTIMEFRAMES --max-tf 0 --input-conf rawAll.cfg | "
fi

#Decoder workflows
if [ $CTFINPUT == 0 ]; then
  WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --pipeline its-stf-decoder:$N_ITSDEC | "
  WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --runmft true | "
  WORKFLOW+="o2-ft0-flp-dpl-workflow $ARGS_ALL --disable-root-output | "
  WORKFLOW+="o2-fv0-flp-dpl-workflow $ARGS_ALL --disable-root-output | "
  WORKFLOW+="o2-mid-raw-to-digits-workflow $ARGS_ALL | "
  WORKFLOW+="o2-tof-compressor $ARGS_ALL | "
  WORKFLOW+="o2-fdd-flp-dpl-workflow --disable-root-output $ARGS_ALL | "
  WORKFLOW+="o2-trd-datareader $ARGS_ALL | "
fi

# Common workflows
WORKFLOW+="o2-its-reco-workflow $ARGS_ALL --trackerCA $DISABLE_MC --clusters-from-upstream --disable-root-output $ITS_CONFIG --configKeyValues \"$ITS_CONFIG_KEY\" | "
WORKFLOW+="o2-gpu-reco-workflow ${ARGS_ALL/--severity $SEVERITY/--severity $SEVERITY_TPC} --input-type=$GPU_INPUT $DISABLE_MC --output-type $GPU_OUTPUT --pipeline gpu-reconstruction:$NGPUS $GPU_CONFIG --configKeyValues \"GPU_global.deviceType=$GPUTYPE;GPU_proc.debugLevel=0;$GPU_CONFIG_KEY;$GPU_EXTRA_CONFIG\" | "
WORKFLOW+="o2-tpcits-match-workflow $ARGS_ALL --disable-root-input --disable-root-output $DISABLE_MC --pipeline itstpc-track-matcher:$N_TPCITS | "
WORKFLOW+="o2-ft0-reco-workflow $ARGS_ALL --disable-root-input --disable-root-output $DISABLE_MC | "
WORKFLOW+="o2-tof-reco-workflow $ARGS_ALL --configKeyValues \"HBFUtils.nHBFPerTF=$NHBPERTF\" --input-type $TOF_INPUT --output-type $TOF_OUTPUT --disable-root-input --disable-root-output $DISABLE_MC | "
WORKFLOW+="o2-trd-tracklet-transformer $ARGS_ALL --root-in 0 --root-out 0 | "
WORKFLOW+="o2-trd-global-tracking $ARGS_ALL --disable-root-input --disable-root-output $TRD_CONFIG | "

# Workflows disabled in sync mode
if [ $SYNCMODE == 0 ]; then
  WORKFLOW+="o2-tof-matcher-tpc $ARGS_ALL --disable-root-input --disable-root-output $DISABLE_MC | "
  WORKFLOW+="o2-mid-reco-workflow $ARGS_ALL --disable-root-output $DISABLE_MC | "
  WORKFLOW+="o2-mft-reco-workflow $ARGS_ALL --clusters-from-upstream $DISABLE_MC --disable-root-output | "
  WORKFLOW+="o2-primary-vertexing-workflow $ARGS_ALL $DISABLE_MC --disable-root-input --disable-root-output --validate-with-ft0 | "
  WORKFLOW+="o2-secondary-vertexing-workflow $ARGS_ALL --disable-root-input --disable-root-output | "
  WORKFLOW+="o2-fdd-reco-workflow $ARGS_ALL --disable-root-input --disable-root-output | "
fi

# Workflows disabled in async mode
if [ $CTFINPUT == 0 ]; then
  WORKFLOW+="o2-phos-reco-workflow $ARGS_ALL --input-type raw --output-type cells --disable-root-input --disable-root-output $DISABLE_MC | "
  WORKFLOW+="o2-cpv-reco-workflow $ARGS_ALL --input-type raw --output-type clusters --disable-root-input --disable-root-output $DISABLE_MC --pipeline CPVClusterizerSpec:$N_CPV | "
  WORKFLOW+="o2-emcal-reco-workflow $ARGS_ALL --input-type raw --output-type cells --disable-root-output $DISABLE_MC --pipeline EMCALRawToCellConverterSpec:$N_EMC | "
  WORKFLOW+="o2-zdc-raw2digits $ARGS_ALL --disable-root-output | "
  WORKFLOW+="o2-hmpid-raw-to-digits-stream-workflow $ARGS_ALL | "

  WORKFLOW+="o2-itsmft-entropy-encoder-workflow $ARGS_ALL --runmft true | "
  WORKFLOW+="o2-ft0-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-fv0-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-mid-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-phos-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-cpv-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-emcal-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-zdc-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-fdd-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-hmpid-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-trd-entropy-encoder-workflow $ARGS_ALL | "
  WORKFLOW+="o2-tpc-reco-workflow --input-type compressed-clusters-flat --output-type encoded-clusters,disable-writer --pipeline tpc-entropy-encoder:$N_TPCENT $ARGS_ALL | "

  WORKFLOW+="o2-tpc-scdcalib-interpolation-workflow $ARGS_ALL --disable-root-output --disable-root-input | "

  # Output workflow
  if [ ! -z $CTF_DIR ] ; then mkdir -p $CTF_DIR; fi
  if [ $CREATECTFDICT == 1 ] ; then
     _DICT_="ctf_dictionary.root"
     if [ ! -z $CTF_DICT_DIR ] ; then
        mkdir -p $CTF_DICT_DIR;
        _DICT_="$CTF_DICT_DIR/$_DICT_"
     fi
     if [ -e $_DICT_ ] ; then rm -f $_DICT_; fi
  fi
  CTF_OUTPUT_TYPE="none"
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="both"; fi
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 0 ]; then CTF_OUTPUT_TYPE="dict"; fi
  if [ $CREATECTFDICT == 0 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="ctf"; fi
  CMD_CTF="o2-ctf-writer-workflow $ARGS_ALL --output-type $CTF_OUTPUT_TYPE --onlyDet $CTF_DETECTORS"
  if [ $CREATECTFDICT == 1 ] && [ $EXTINPUT == 1 ]; then CMD_CTF+=" --save-dict-after $NTIMEFRAMES"; fi
  if [ ! -z $CTF_DICT_DIR ]; then CMD_CTF+=" --ctf-dict-dir $CTF_DICT_DIR"; fi
  if [ ! -z $CTF_DIR ]; then CMD_CTF+=" --ctf-output-dir $CTF_DIR"; fi
  WORKFLOW+="$CMD_CTF | "
fi

# DPL run binary
WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT --run"

if [ "0$PRINT_WORKFLOW_ONLY" == "01" ]; then
  echo Workflow command:
  echo $WORKFLOW | sed "s/| */|\n/g"
else
  # Execute the command we have assembled
  eval $WORKFLOW
fi
