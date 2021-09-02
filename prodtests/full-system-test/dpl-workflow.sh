#!/bin/bash

# In order to use o2sim_grp.root, o2sim_geometry.root or matbud.root from arbitrary directory one can provide to the workflow
# --configKeyValues "NameConf.mDirGRP=<dirname1>;NameConf.mDirGeom=<dirname2>;NameConf.mDirMatLUT=<dirname3>;"
# All workflows currently running in the FST parce the configKeyValues option, so the NameConf.mDirXXX keys can be added via global env.var.

# Get this script's directory : use zsh first (e.g. on Mac) and fallback
# on `readlink -f` if zsh is not there
command -v zsh > /dev/null 2>&1 && MYDIR=$(dirname $(zsh -c 'echo ${0:A}' "$0"))
test -z ${MYDIR+x} && MYDIR="$(dirname $(readlink -f $0))"

source $MYDIR/setenv.sh

if [ -z $OPTIMIZED_PARALLEL_ASYNC ]; then OPTIMIZED_PARALLEL_ASYNC=0; fi

workflow_has_parameter CTF && export SAVECTF=1
workflow_has_parameter GPU && { export GPUTYPE=HIP; export NGPUS=4; }

if [ "0$O2_ROOT" == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load O2/latest
fi

# Set general arguments
ARGS_ALL="--session default --severity $SEVERITY --shm-segment-id $NUMAID --shm-segment-size $SHMSIZE"
if [ $EPNMODE == 1 ]; then
  ARGS_ALL+=" --infologger-severity $INFOLOGGER_SEVERITY"
  #ARGS_ALL+=" --monitoring-backend influxdb-unix:///tmp/telegraf.sock"
  ARGS_ALL+=" --monitoring-backend no-op://"
else
  ARGS_ALL+=" --monitoring-backend no-op://"
fi
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
if [ $GPUTYPE != "CPU" ] || [ $OPTIMIZED_PARALLEL_ASYNC != 0 ]; then
  ARGS_ALL+="  --shm-mlock-segment-on-creation 1"
fi
ARGS_ALL_CONFIG="NameConf.mDirGRP=$FILEWORKDIR;NameConf.mDirGeom=$FILEWORKDIR;NameConf.mDirCollContext=$FILEWORKDIR;NameConf.mDirMatLUT=$FILEWORKDIR;keyval.input_dir=$FILEWORKDIR;keyval.output_dir=/dev/null"

# Set some individual workflow arguments depending on configuration
if [ -z $CTF_DIR ]; then CTF_DIR=$FILEWORKDIR; fi
if [ -z $CTF_DICT_DIR ]; then CTF_DICT_DIR=$FILEWORKDIR; fi
GPU_INPUT=zsraw
GPU_OUTPUT=tracks,clusters
GPU_CONFIG=
GPU_CONFIG_KEY=
TOF_INPUT=raw
TOF_OUTPUT=clusters
ITS_CONFIG=
ITS_CONFIG_KEY=
TRD_CONFIG=
TRD_CONFIG_KEY=
TRD_TRANSFORMER_CONFIG=
EVE_CONFIG=
if [ $SYNCMODE == 1 ]; then
  ITS_CONFIG_KEY+="fastMultConfig.cutMultClusLow=30;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500;"
  GPU_CONFIG_KEY+="GPU_global.synchronousProcessing=1;GPU_proc.clearO2OutputFromGPU=1;"
  TRD_CONFIG+=" --track-sources ITS-TPC --filter-trigrec"
  TRD_CONFIG_KEY+="GPU_proc.ompThreads=1;"
  TRD_TRANSFORMER_CONFIG+=" --filter-trigrec"
else
  TRD_CONFIG+=" --track-sources TPC,ITS-TPC"
fi
if [ $CTFINPUT == 1 ]; then
  ITS_CONFIG+=" --tracking-mode async"
else
  ITS_CONFIG+=" --entropy-encoding"
  TOF_OUTPUT+=",ctf"
  GPU_OUTPUT+=",compressed-clusters-ctf"
fi
if [ $EPNMODE == 1 ]; then
  EVE_CONFIG+=" --eve-dds-collection-index 0"
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

if [ ! -z $GPU_NUM_MEM_REG_CALLBACKS ]; then
  GPU_CONFIG+=" --expected-region-callbacks $GPU_NUM_MEM_REG_CALLBACKS"
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

N_TPCTRK=1
N_TPCENT=1
N_TPCITS=1
N_ITSDEC=1
N_EMC=1
N_TRDENT=1
N_TRDTRK=1
N_TPCENTDEC=1
N_MFTTRK=1
N_ITSTRK=1
N_MCHTRK=1
N_TOFMATCH=1
if [ $OPTIMIZED_PARALLEL_ASYNC != 0 ]; then
  if [ $SYNCMODE == "1" ]; then echo "Must not use OPTIMIZED_PARALLEL_ASYNC with GPU or SYNCMODE"; exit 1; fi
  if [ $NUMAGPUIDS == 1 ]; then N_NUMAFACTOR=1; else N_NUMAFACTOR=2; fi
  GPU_CONFIG_KEY+="GPU_proc.ompThreads=6;"
  TRD_CONFIG_KEY+="GPU_proc.ompThreads=2;"
  if [ $GPUTYPE == "CPU" ]; then
    N_TPCENTDEC=$(expr 2 \* $N_NUMAFACTOR)
    N_MFTTRK=$(expr 3 \* $N_NUMAFACTOR)
    N_ITSTRK=$(expr 3 \* $N_NUMAFACTOR)
    N_TPCITS=$(expr 2 \* $N_NUMAFACTOR)
    N_MCHTRK=$(expr 1 \* $N_NUMAFACTOR)
    N_TOFMATCH=$(expr 9 \* $N_NUMAFACTOR)
    N_TPCTRK=$(expr 6 \* $N_NUMAFACTOR)
  else
    N_TPCENTDEC=$((3 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 > 0 ? 3 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 : 1))
    N_MFTTRK=$((6 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 > 0 ? 6 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 : 1))
    N_ITSTRK=$((6 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 > 0 ? 6 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 : 1))
    N_TPCITS=$((4 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 > 0 ? 4 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 : 1))
    N_MCHTRK=$((2 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 > 0 ? 2 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 : 1))
    N_TOFMATCH=$((20 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 > 0 ? 20 * $NGPUS * $OPTIMIZED_PARALLEL_ASYNC * $N_NUMAFACTOR / 4 : 1))
    N_TPCTRK=$NGPUS
  fi
elif [ $EPNPIPELINES != 0 ]; then
  N_TPCENT=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_TPCITS=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_ITSDEC=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_EMC=$(($(expr 7 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 7 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_TRDENT=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  N_TRDTRK=$(($(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) > 0 ? $(expr 3 \* $EPNPIPELINES \* $NGPUS / 4) : 1))
  if [ $GPUTYPE == "CPU" ]; then
    N_TPCTRK=8
    GPU_CONFIG_KEY+="GPU_proc.ompThreads=4;"
  else
    N_TPCTRK=$NGPUS
  fi
fi

# Input workflow
if [ $CTFINPUT == 1 ]; then
  GPU_INPUT=compressed-clusters-ctf
  TOF_INPUT=digits
  CTFName=`ls -t $FILEWORKDIR/o2_ctf_*.root | head -n1`
  WORKFLOW="o2-ctf-reader-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --delay $TFDELAY --loop $NTIMEFRAMES --ctf-input ${CTFName} --ctf-dict ${CTF_DICT_DIR}/ctf_dictionary.root --onlyDet $WORKFLOW_DETECTORS --pipeline tpc-entropy-decoder:$N_TPCENTDEC | "
elif [ $EXTINPUT == 1 ]; then
  WORKFLOW="o2-dpl-raw-proxy $ARGS_ALL --dataspec \"FLP:FLP/DISTSUBTIMEFRAME/0;B:TPC/RAWDATA;C:ITS/RAWDATA;D:TOF/RAWDATA;D:MFT/RAWDATA;E:FT0/RAWDATA;F:MID/RAWDATA;G:EMC/RAWDATA;H:PHS/RAWDATA;I:CPV/RAWDATA;J:ZDC/RAWDATA;K:HMP/RAWDATA;L:FDD/RAWDATA;M:TRD/RAWDATA;N:FV0/RAWDATA;O:MCH/RAWDATA\" --channel-config \"name=readout-proxy,type=pull,method=connect,address=ipc://@$INRAWCHANNAME,transport=shmem,rateLogging=0\" | "
else
  WORKFLOW="o2-raw-file-reader-workflow --detect-tf0 $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG;HBFUtils.nHBFPerTF=$NHBPERTF;\" --delay $TFDELAY --loop $NTIMEFRAMES --max-tf 0 --input-conf $FILEWORKDIR/rawAll.cfg | "
fi

if [ $EPNMODE == 1 ]; then
  GPU_INPUT=zsonthefly
  WORKFLOW+="o2-tpc-raw-to-digits-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-spec \"A:TPC/RAWDATA;dd:FLP/DISTSUBTIMEFRAME/0\" --remove-duplicates --pipeline tpc-raw-to-digits-0:6 | "
  WORKFLOW+="o2-tpc-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type digitizer --output-type zsraw,disable-writer --pipeline tpc-zsEncoder:6 | "
fi

#Decoder workflows
if [ $CTFINPUT == 0 ]; then
  has_detector ITS && WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --dict-file $FILEWORKDIR/ITSdictionary.bin --pipeline its-stf-decoder:$N_ITSDEC | "
  has_detector MFT && WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --dict-file $FILEWORKDIR/MFTdictionary.bin --runmft true | "
  has_detector FT0 && [ $EPNMODE == 0 ] && WORKFLOW+="o2-ft0-flp-dpl-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output | "
  has_detector FV0 && [ $EPNMODE == 0 ] && WORKFLOW+="o2-fv0-flp-dpl-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output | "
  has_detector MID && [ $EPNMODE == 0 ] && WORKFLOW+="o2-mid-raw-to-digits-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector MCH && WORKFLOW+="o2-mch-raw-to-digits-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector TOF && [ $EPNMODE == 0 ] && WORKFLOW+="o2-tof-compressor $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector FDD && [ $EPNMODE == 0 ] && WORKFLOW+="o2-fdd-flp-dpl-workflow --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output $ARGS_ALL | "
  has_detector TRD && WORKFLOW+="o2-trd-datareader $ARGS_ALL | "
  has_detector ZDC && WORKFLOW+="o2-zdc-raw2digits $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output | "
  has_detector HMP && WORKFLOW+="o2-hmpid-raw-to-digits-stream-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
fi

# Common workflows
has_detector ITS && WORKFLOW+="o2-its-reco-workflow $ARGS_ALL --trackerCA $DISABLE_MC --clusters-from-upstream --disable-root-output $ITS_CONFIG --configKeyValues \"$ARGS_ALL_CONFIG;$ITS_CONFIG_KEY\" --its-dictionary-path $FILEWORKDIR --pipeline its-tracker:$N_ITSTRK | "
has_detector TPC && WORKFLOW+="o2-gpu-reco-workflow ${ARGS_ALL//-severity $SEVERITY/-severity $SEVERITY_TPC} --input-type=$GPU_INPUT $DISABLE_MC --output-type $GPU_OUTPUT --pipeline gpu-reconstruction:$N_TPCTRK $GPU_CONFIG --configKeyValues \"$ARGS_ALL_CONFIG;GPU_global.deviceType=$GPUTYPE;GPU_proc.debugLevel=0;$GPU_CONFIG_KEY;$GPU_EXTRA_CONFIG\" | "
has_detectors ITS TPC && WORKFLOW+="o2-tpcits-match-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --its-dictionary-path $FILEWORKDIR --pipeline itstpc-track-matcher:$N_TPCITS | "
has_detector FT0 && WORKFLOW+="o2-ft0-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC | "
has_detector TOF && WORKFLOW+="o2-tof-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type $TOF_INPUT --output-type $TOF_OUTPUT --disable-root-input --disable-root-output $DISABLE_MC | "
has_detector TRD && WORKFLOW+="o2-trd-tracklet-transformer $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC $TRD_TRANSFORMER_CONFIG --pipeline TRDTRACKLETTRANSFORMER:$N_TRDTRK | "
has_detectors TRD TPC ITS && WORKFLOW+="o2-trd-global-tracking $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG;$TRD_CONFIG_KEY\" --disable-root-input --disable-root-output $DISABLE_MC $TRD_CONFIG | "

# Workflows disabled in sync mode
if [ $SYNCMODE == 0 ]; then
  has_detectors TOF TRD TPC ITS && WORKFLOW+="o2-tof-matcher-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --track-sources \"TPC,ITS-TPC\" --pipeline tof-matcher:$N_TOFMATCH | "
  has_detector MID && WORKFLOW+="o2-mid-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output $DISABLE_MC | "
  has_detector MCH && WORKFLOW+="o2-mch-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --pipeline mch-track-finder:$N_MCHTRK | "
  has_detector MFT && WORKFLOW+="o2-mft-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --clusters-from-upstream $DISABLE_MC --disable-root-output --pipeline mft-tracker:$N_MFTTRK --mft-dictionary-path $FILEWORKDIR | "
  has_detectors ITS TPC TRD TOF FT0 MCH && WORKFLOW+="o2-primary-vertexing-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" $DISABLE_MC --disable-root-input --disable-root-output --validate-with-ft0 | "
  has_detectors ITS TPC TRD TOF FT0 MCH && WORKFLOW+="o2-secondary-vertexing-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output | "
  has_detector FDD && WORKFLOW+="o2-fdd-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output | "
  has_detector ZDC && WORKFLOW+="o2-zdc-digits-reco $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output  $DISABLE_MC | "
fi

# Workflows disabled in async mode
if [ $CTFINPUT == 0 ]; then
  if [ $SYNCMODE == 1 ]; then # Otherwise already present in async setup
    has_detectors TOF TRD TPC ITS && WORKFLOW+="o2-tof-matcher-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --track-sources \"ITS-TPC\" | "
  fi
  has_detector PHS && WORKFLOW+="o2-phos-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type raw --output-type cells --disable-root-input --disable-root-output $DISABLE_MC | "
  has_detector CPV && WORKFLOW+="o2-cpv-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type raw --output-type clusters --disable-root-input --disable-root-output $DISABLE_MC | "
  has_detector EMC && WORKFLOW+="o2-emcal-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type raw --output-type cells --disable-root-output $DISABLE_MC --pipeline EMCALRawToCellConverterSpec:$N_EMC | "

  has_detector MFT && WORKFLOW+="o2-itsmft-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --runmft true | "
  has_detector FT0 && WORKFLOW+="o2-ft0-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector FV0 && WORKFLOW+="o2-fv0-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector MID && WORKFLOW+="o2-mid-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector MCH && WORKFLOW+="o2-mch-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector PHS && WORKFLOW+="o2-phos-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector CPV && WORKFLOW+="o2-cpv-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector EMC && WORKFLOW+="o2-emcal-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector ZDC && WORKFLOW+="o2-zdc-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector FDD && WORKFLOW+="o2-fdd-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector HMP && WORKFLOW+="o2-hmpid-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector TRD && WORKFLOW+="o2-trd-entropy-encoder-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline trd-entropy-encoder:$N_TRDENT | "
  has_detector TPC && WORKFLOW+="o2-tpc-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type compressed-clusters-flat --output-type encoded-clusters,disable-writer --pipeline tpc-entropy-encoder:$N_TPCENT | "

  has_detectors TPC ITS TRD TOF && WORKFLOW+="o2-tpc-scdcalib-interpolation-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output --disable-root-input | "

  # Output workflow
  if [ $SAVECTF == 1 ]; then
    mkdir -p $CTF_DIR
  fi
  if [ $CREATECTFDICT == 1 ] ; then
    mkdir -p $CTF_DICT_DIR;
    rm -f $CTF_DICT_DIR/ctf_dictionary.root
  fi
  CTF_OUTPUT_TYPE="none"
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="both"; fi
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 0 ]; then CTF_OUTPUT_TYPE="dict"; fi
  if [ $CREATECTFDICT == 0 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="ctf"; fi
  CMD_CTF="o2-ctf-writer-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --output-dir $CTF_DIR --ctf-dict-dir $CTF_DICT_DIR --output-type $CTF_OUTPUT_TYPE --onlyDet $WORKFLOW_DETECTORS"
  if [ $CREATECTFDICT == 1 ] && [ $EXTINPUT == 1 ]; then CMD_CTF+=" --save-dict-after $NTIMEFRAMES"; fi
  WORKFLOW+="$CMD_CTF | "
fi

workflow_has_parameter EVENT_DISPLAY && [ $NUMAID == 0 ] && WORKFLOW+="o2-eve-display $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --display-tracks TPC --display-clusters TPC $EVE_CONFIG $DISABLE_MC | "

workflow_has_parameter QC && source $MYDIR/qc-workflow.sh

# DPL run binary
WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT"

if [ $WORKFLOWMODE == "print" ]; then
  echo Workflow command:
  echo $WORKFLOW | sed "s/| */|\n/g"
else
  # Execute the command we have assembled
  WORKFLOW+=" --$WORKFLOWMODE"
  eval $WORKFLOW
fi
