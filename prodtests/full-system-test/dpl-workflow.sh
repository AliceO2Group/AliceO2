#!/bin/bash

# Non-zero exit code already if one command in a pipe fails
set -o pipefail

# Abort in case any variable is not bound
if [[ ${IGNORE_UNBOUND_VARIABLES:-} != 1 ]]; then  set -u; fi

# ---------------------------------------------------------------------------------------------------------------------
# Get this script's directory and load common settings
: ${GEN_TOPO_MYDIR:=$(dirname $(realpath $0))}
source $GEN_TOPO_MYDIR/gen_topo_helper_functions.sh || { echo "gen_topo_helper_functions.sh failed" 1>&2 && exit 1; }
source $GEN_TOPO_MYDIR/setenv.sh || { echo "setenv.sh failed" 1>&2 && exit 1; }

if [[ $EPNSYNCMODE == 0 && ${DPL_CONDITION_BACKEND:-} != "http://o2-ccdb.internal" && ${DPL_CONDITION_BACKEND:-} != "http://localhost:8084" && ${DPL_CONDITION_BACKEND:-} != "http://127.0.0.1:8084" ]]; then
  alien-token-info >& /dev/null
  if [[ $? != 0 ]]; then
    echo "FATAL: No alien token present" 1>&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------------------------------------------------
#Some additional settings used in this workflow
: ${CTF_DICT:="ctf_dictionary.root"}  # Local dictionary file name if its creation is request
: ${RECO_NUM_NODES_WORKFLOW:="230"}   # Number of EPNs running this workflow in parallel, to increase multiplicities if necessary, by default assume we are 1 out of 250 servers
: ${CTF_MINSIZE:="10000000000"}       # accumulate CTFs until file size reached
: ${CTF_MAX_PER_FILE:="40000"}        # but no more than given number of CTFs per file
: ${CTF_FREE_DISK_WAIT:="10"}         # if disk on EPNs is close to full, wait X seconds before retrying to write
: ${CTF_MAX_FREE_DISK_WAIT:="600"}    # if not enough disk space after this time throw error

workflow_has_parameter CTF && export SAVECTF=1
workflow_has_parameter GPU && { export GPUTYPE=HIP; export NGPUS=4; }

# ---------------------------------------------------------------------------------------------------------------------
# Set general arguments
source $GEN_TOPO_MYDIR/getCommonArgs.sh || { echo "getCommonArgs.sh failed" 1>&2 && exit 1; }
workflow_has_parameter CALIB && { source $O2DPG_ROOT/DATA/common/setenv_calib.sh; [[ $? != 0 ]] && echo "setenv_calib.sh failed" 1>&2 && exit 1; }

[[ -z ${SHM_MANAGER_SHMID:-} ]] && ( [[ $EXTINPUT == 1 ]] || [[ $NUMAGPUIDS != 0 ]] ) && ARGS_ALL+=" --no-cleanup"
[[ $GPUTYPE != "CPU" || ( ! -z ${OPTIMIZED_PARALLEL_ASYNC:-} && -z ${SETENV_NO_ULIMIT:-} ) ]] && ARGS_ALL+=" --shm-mlock-segment-on-creation 1"
if [[ $EPNSYNCMODE == 1 ]] || type numactl >/dev/null 2>&1 && [[ `numactl -H | grep "node . size" | wc -l` -ge 2 ]]; then
  [[ $NUMAGPUIDS != 0 ]] && ARGS_ALL+=" --child-driver 'numactl --membind $NUMAID --cpunodebind $NUMAID'"
fi
if [[ -z ${TIMEFRAME_RATE_LIMIT:-} ]] && [[ $DIGITINPUT != 1 ]]; then
  TIMEFRAME_RATE_LIMIT=$((12 * 230 / $RECO_NUM_NODES_WORKFLOW * ($NUMAGPUIDS != 0 ? 1 : 2) * 128 / $NHBPERTF))
  [[ $BEAMTYPE != "PbPb" && ${HIGH_RATE_PP:-0} == 0 ]] && TIMEFRAME_RATE_LIMIT=$(($TIMEFRAME_RATE_LIMIT * 3))
  ! has_detector TPC && TIMEFRAME_RATE_LIMIT=$(($TIMEFRAME_RATE_LIMIT * 4))
  [[ ! -z ${EPN_GLOBAL_SCALING:-} ]] && TIMEFRAME_RATE_LIMIT=$(($TIMEFRAME_RATE_LIMIT * $EPN_GLOBAL_SCALING))
fi
[[ ! -z ${TIMEFRAME_RATE_LIMIT:-} ]] && [[ $TIMEFRAME_RATE_LIMIT != 0 ]] && ARGS_ALL+=" --timeframes-rate-limit $TIMEFRAME_RATE_LIMIT --timeframes-rate-limit-ipcid ${O2JOBID:-$NUMAID}"
if [[ $EPNSYNCMODE == 1 ]]; then
  SYNCRAWMODE=1
elif [[ -z ${SYNCRAWMODE:-} ]]; then
  SYNCRAWMODE=0
fi

# ---------------------------------------------------------------------------------------------------------------------
# Process multiplicities

{ source $O2DPG_ROOT/DATA/production/workflow-multiplicities.sh; [[ $? != 0 ]] && echo "workflow-multiplicities.sh failed" 1>&2 && exit 1; }

# ---------------------------------------------------------------------------------------------------------------------
# Set some individual workflow arguments depending on configuration
GPU_INPUT=zsraw
GPU_OUTPUT=tracks,clusters
GPU_CONFIG=
GPU_CONFIG_KEY=
TOF_CONFIG=
TOF_INPUT=raw
TOF_OUTPUT=clusters
ITS_CONFIG_KEY=
MFT_CONFIG=
MFT_CONFIG_KEY=
TRD_CONFIG=
TRD_CONFIG_KEY=
TRD_FILTER_CONFIG=
CPV_INPUT=raw
EVE_CONFIG=" --jsons-folder $EDJSONS_DIR"
MIDDEC_CONFIG=
EMCRAW2C_CONFIG=
PHS_CONFIG=
MCH_CONFIG_KEY=
CTP_CONFIG=
: ${ITSEXTRAERR:=}
: ${TRACKTUNETPCINNER:=}
: ${ITSTPC_CONFIG_KEY:=}

[[ "0$DISABLE_ROOT_OUTPUT" == "00" ]] && DISABLE_ROOT_OUTPUT=

if [[ -z ${ALPIDE_ERR_DUMPS:-} ]]; then
  [[ $EPNSYNCMODE == 1 ]] && ALPIDE_ERR_DUMPS="1" || ALPIDE_ERR_DUMPS="0"
fi

if [[ $SYNCMODE == 1 ]]; then
  if [[ $BEAMTYPE == "PbPb" ]]; then
    ITS_CONFIG_KEY+="fastMultConfig.cutMultClusLow=30;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500;"
    MCH_CONFIG_KEY="MCHTracking.maxCandidates=50000;MCHTracking.maxTrackingDuration=20;"
  elif [[ $BEAMTYPE == "pp" ]]; then
    ITS_CONFIG_KEY+="fastMultConfig.cutMultClusLow=-1;fastMultConfig.cutMultClusHigh=-1;fastMultConfig.cutMultVtxHigh=-1;ITSVertexerParam.phiCut=0.5;ITSVertexerParam.clusterContributorsCut=3;ITSVertexerParam.tanLambdaCut=0.2;"
    [[ ! -z ${CUT_RANDOM_FRACTION_ITS:-} ]] && ITS_CONFIG_KEY+="fastMultConfig.cutRandomFraction=$CUT_RANDOM_FRACTION_ITS;"
    MCH_CONFIG_KEY="MCHTracking.maxCandidates=20000;MCHTracking.maxTrackingDuration=10;"
  fi
  if has_detector_reco ITS; then
    [[ $RUNTYPE == "COSMICS" ]] && MFT_CONFIG_KEY+="MFTTracking.irFramesOnly=1;"
  else
    MFT_CONFIG_KEY+="MFTTracking.cutMultClusLow=0;MFTTracking.cutMultClusHigh=2000;"
  fi

  PVERTEXING_CONFIG_KEY+="pvertexer.meanVertexExtraErrConstraint=0.3;" # for calibration relax the constraint
  if [[ $SYNCRAWMODE == 1 ]]; then # add extra tolerance in sync mode to account for eventual time misalignment
    PVERTEXING_CONFIG_KEY+="pvertexer.timeMarginVertexTime=1.3;"
    if [[ -z $ITSEXTRAERR ]]; then # in sync mode account for ITS residual misalignment
      ERRIB="100e-8"
      ERROB="100e-8"
      ITSEXTRAERR="ITSCATrackerParam.sysErrY2[0]=$ERRIB;ITSCATrackerParam.sysErrZ2[0]=$ERRIB;ITSCATrackerParam.sysErrY2[1]=$ERRIB;ITSCATrackerParam.sysErrZ2[1]=$ERRIB;ITSCATrackerParam.sysErrY2[2]=$ERRIB;ITSCATrackerParam.sysErrZ2[2]=$ERRIB;ITSCATrackerParam.sysErrY2[3]=$ERROB;ITSCATrackerParam.sysErrZ2[3]=$ERROB;ITSCATrackerParam.sysErrY2[4]=$ERROB;ITSCATrackerParam.sysErrZ2[4]=$ERROB;ITSCATrackerParam.sysErrY2[5]=$ERROB;ITSCATrackerParam.sysErrZ2[5]=$ERROB;ITSCATrackerParam.sysErrY2[6]=$ERROB;ITSCATrackerParam.sysErrZ2[6]=$ERROB;"
    fi
    ITSTPC_CONFIG_KEY+="tpcitsMatch.safeMarginTimeCorrErr=5.;tpcitsMatch.cutMatchingChi2=60;"
    if [[ -z $TRACKTUNETPCINNER ]]; then # account for extra TPC errors
      TRACKTUNETPCINNER="trackTuneParams.sourceLevelTPC=true;trackTuneParams.tpcCovInnerType=1;trackTuneParams.useTPCInnerCorr=false;trackTuneParams.tpcCovInner[0]=0.25;trackTuneParams.tpcCovInner[2]=2.25e-4;trackTuneParams.tpcCovInner[3]=2.25e-4;trackTuneParams.tpcCovInner[4]=0.0256;"
    fi
  fi
  GPU_CONFIG_KEY+="GPU_global.synchronousProcessing=1;GPU_proc.clearO2OutputFromGPU=1;"
  has_processing_step TPC_DEDX && GPU_CONFIG_KEY+="GPU_global.rundEdx=1;"
  has_detector ITS && TRD_FILTER_CONFIG+=" --filter-trigrec"
else
  if [[ $BEAMTYPE == "pp" ]]; then
    ITS_CONFIG_KEY+="ITSVertexerParam.phiCut=0.5;ITSVertexerParam.clusterContributorsCut=3;ITSVertexerParam.tanLambdaCut=0.2;"
  fi
fi
[[ $CTFINPUT == 1 ]] && GPU_CONFIG_KEY+="GPU_proc.tpcInputWithClusterRejection=1;"
[[ ! -z $NTRDTRKTHREADS ]] && TRD_CONFIG_KEY+="GPU_proc.ompThreads=$NTRDTRKTHREADS;"
[[ ! -z $NGPURECOTHREADS ]] && GPU_CONFIG_KEY+="GPU_proc.ompThreads=$NGPURECOTHREADS;"
[[ ! -z $NMFTTHREADS ]] && MFT_CONFIG+=" --nThreads $NMFTTHREADS"
[[ $ITSTRK_THREADS != 1 ]] && ITS_CONFIG_KEY+="ITSVertexerParam.nThreads=$ITSTRK_THREADS;ITSCATrackerParam.nThreads=$ITSTRK_THREADS;"

if [[ $BEAMTYPE == "PbPb" ]]; then
  PVERTEXING_CONFIG_KEY+="pvertexer.maxChi2TZDebris=2000;"
elif [[ $BEAMTYPE == "pp" ]]; then
  PVERTEXING_CONFIG_KEY+="pvertexer.maxChi2TZDebris=10;"
fi

if [[ $BEAMTYPE == "cosmic" ]]; then
  [[ -z ${ITS_CONFIG+x} ]] && ITS_CONFIG=" --tracking-mode cosmics"
elif [[ $SYNCMODE == 1 ]]; then
  [[ -z ${ITS_CONFIG+x} ]] && ITS_CONFIG=" --tracking-mode sync"
else
  [[ -z ${ITS_CONFIG+x} ]] && ITS_CONFIG=" --tracking-mode async"
fi

if [[ $SYNCMODE == 1 ]]; then
  if has_detector TRD && [[ ! -z ${PRESCALE_ITS_WITH_TRD:-} ]]; then
    ITS_CONFIG+=" --select-with-triggers trd "
  else
    ITS_CONFIG+=" --select-with-triggers phys "
  fi
fi


workflow_has_parameter CALIB && [[ $CALIB_TRD_VDRIFTEXB == 1 ]] && TRD_CONFIG+=" --enable-trackbased-calib"
! has_detector FT0 && TRD_CONFIG+=" --disable-ft0-pileup-tagging"

SEND_ITSTPC_DTGL=
workflow_has_parameter CALIB && [[ $CALIB_TPC_VDRIFTTGL == 1 ]] && SEND_ITSTPC_DTGL="--produce-calibration-data"

PVERTEXING_CONFIG_KEY+="${ITSMFT_STROBES};"

has_processing_step ENTROPY_ENCODER && has_detector_ctf TPC && GPU_OUTPUT+=",compressed-clusters-ctf"

if [[ $SYNCMODE == 1 ]] && workflow_has_parameter QC && has_detector_qc TPC; then
  GPU_OUTPUT+=",qa,error-qa"
  [[ -z ${TPC_TRACKING_QC_RUN_FRACTION:-} ]] && TPC_TRACKING_QC_RUN_FRACTION=1
  GPU_CONFIG_KEY+="GPU_QA.clusterRejectionHistograms=1;GPU_proc.qcRunFraction=$TPC_TRACKING_QC_RUN_FRACTION;"
  [[ $GPUTYPE != "CPU" && $HOSTMEMSIZE == "0" && $TPC_TRACKING_QC_RUN_FRACTION == "100" ]] && HOSTMEMSIZE=$(( 5 << 30 ))
fi

# enable only if root output is written, because it slows down the processing
[[ -z $DISABLE_ROOT_OUTPUT ]] && ENABLE_ROOT_OUTPUT="--enable-root-output"
[[ -z $DISABLE_ROOT_OUTPUT ]] || needs_root_output o2-gpu-reco-workflow && GPU_OUTPUT+=",send-clusters-per-sector"

has_detector_flp_processing CPV && CPV_INPUT=digits
! has_detector_flp_processing TOF && TOF_CONFIG+=" --local-cmp"

if [[ $EPNSYNCMODE == 1 ]]; then
  EVE_CONFIG+=" --eve-dds-collection-index 0"
  MIDDEC_CONFIG+=" --feeId-config-file \"$MID_FEEID_MAP\""
  if [[ $EXTINPUT == 1 ]] && [[ $GPUTYPE != "CPU" ]] && [[ -z "$GPU_NUM_MEM_REG_CALLBACKS" ]]; then GPU_NUM_MEM_REG_CALLBACKS=4; fi
fi
if [[ $SYNCRAWMODE == 1 ]]; then
  GPU_CONFIG_KEY+="GPU_proc.tpcIncreasedMinClustersPerRow=500000;GPU_proc.ignoreNonFatalGPUErrors=1;GPU_proc.throttleAlarms=1;GPU_proc.conservativeMemoryEstimate=1;"
  # option for avoinding masking problematic channels from previous calibrations
  TOF_CONFIG+=" --for-calib"
fi

if [[ $SYNCMODE == 1 && "0${ED_NO_ITS_ROF_FILTER:-}" != "01" && $BEAMTYPE == "PbPb" ]] && has_detector ITS; then
  EVE_CONFIG+=" --filter-its-rof"
fi

if [[ ! -z ${EVE_NTH_EVENT:-} ]]; then
  EVE_CONFIG+=" --only-nth-event=$EVE_NTH_EVENT"
fi

if [[ $GPUTYPE != "CPU" && $NUMAGPUIDS != 0 ]] && [[ -z ${ROCR_VISIBLE_DEVICES:-} || ${ROCR_VISIBLE_DEVICES:-} = "0,1,2,3,4,5,6,7" || ${ROCR_VISIBLE_DEVICES:-} = "0,1,2,3" || ${ROCR_VISIBLE_DEVICES:-} = "4,5,6,7" ]]; then
  GPU_CONFIG_KEY+="GPU_global.registerSelectedSegmentIds=$NUMAID;"
fi

if [[ $GPUTYPE == "HIP" ]]; then
  GPU_CONFIG_KEY+="GPU_proc.deviceNum=0;"
  if [[ $NGPUS != 1 || $NUMAID != 0 ]]; then
    if [[ -z ${ROCR_VISIBLE_DEVICES:-} ]]; then
      GPU_FIRST_ID=0
    else
      GPU_FIRST_ID=$(echo ${ROCR_VISIBLE_DEVICES//,/ } | awk '{print $1}')
    fi
    TIMESLICEOFFSET=$(($GPU_FIRST_ID + ($NUMAGPUIDS != 0 ? ($NGPUS * $NUMAID) : 0)))
    GPU_CONFIG+=" --environment \"ROCR_VISIBLE_DEVICES={timeslice${TIMESLICEOFFSET}}\""
  fi
  [[ "${EPN_NODE_MI100:-}" != "1" ]] && export HSA_NO_SCRATCH_RECLAIM=1
  #export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
else
  GPU_CONFIG_KEY+="GPU_proc.deviceNum=-2;"
fi

if [[ ! -z ${GPU_NUM_MEM_REG_CALLBACKS:-} ]]; then
  GPU_CONFIG+=" --expected-region-callbacks $GPU_NUM_MEM_REG_CALLBACKS"
fi

if [[ $GPUTYPE != "CPU" ]]; then
  GPU_CONFIG_KEY+="GPU_proc.forceMemoryPoolSize=$GPUMEMSIZE;"
  [[ $HOSTMEMSIZE == "0" ]] && HOSTMEMSIZE=$(( 1 << 30 ))
fi

if [[ $HOSTMEMSIZE != "0" ]]; then
  GPU_CONFIG_KEY+="GPU_proc.forceHostMemoryPoolSize=$HOSTMEMSIZE;"
fi

GPU_CONFIG_SELF="--severity $SEVERITY_TPC"
GPU_CONFIG_SELF+=" $TPC_CORR_SCALING "
if [[ $GPUTYPE != "CPU" && $(ulimit -e) -ge 25 && ${O2_GPU_WORKFLOW_NICE:-} == 1 ]]; then
  GPU_CONFIG_SELF+=" --child-driver 'nice -n -5'"
fi

if ! has_detector_reco TOF; then
  TOF_OUTPUT=digits
elif [[ -z "$DISABLE_ROOT_OUTPUT" ]] && has_detector_reco TOF && ! has_detector_from_global_reader TOF; then
  TOF_OUTPUT+=",digits"
fi

if has_detector_calib PHS && workflow_has_parameter CALIB; then
  PHS_CONFIG+="--fullclu-output"
fi

( workflow_has_parameter AOD || [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-emcal-cell-writer-workflow ) && has_detector EMC && RAW_EMC_SUBSPEC=" --subspecification 1 "
has_detector_reco MID && has_detector_matching MCHMID && MFTMCHConf="FwdMatching.useMIDMatch=true;" || MFTMCHConf="FwdMatching.useMIDMatch=false;"

[[ $IS_SIMULATED_DATA == "1" ]] && EMCRAW2C_CONFIG+=" --no-mergeHGLG"

# ---------------------------------------------------------------------------------------------------------------------
# Temporary extra options

if has_processing_step MUON_SYNC_RECO; then
  [[ -z ${ARGS_EXTRA_PROCESS_o2_mid_reco_workflow:-} ]] && ARGS_EXTRA_PROCESS_o2_mid_reco_workflow="--mid-tracker-keep-best"
  [[ -z ${ARGS_EXTRA_PROCESS_o2_mch_reco_workflow:-} ]] && ARGS_EXTRA_PROCESS_o2_mch_reco_workflow="--digits"
  if [[ -z ${CONFIG_EXTRA_PROCESS_o2_mch_reco_workflow:-} ]]; then
    if [[ $RUNTYPE == "PHYSICS" || $RUNTYPE == "COSMICS" ]]; then
      CONFIG_EXTRA_PROCESS_o2_mch_reco_workflow="MCHTracking.chamberResolutionX=0.4;MCHTracking.chamberResolutionY=0.4;MCHTracking.sigmaCutForTracking=7.;MCHDigitFilter.timeOffset=126;MCHTracking.sigmaCutForImprovement=6.;MCHTracking.maxCandidates=20000;"
    elif [[ $RUNTYPE == "SYNTHETIC" ]]; then
      CONFIG_EXTRA_PROCESS_o2_mch_reco_workflow="MCHTimeClusterizer.peakSearchSignalOnly=false;MCHDigitFilter.rejectBackground=false;MCHTracking.chamberResolutionX=0.4;MCHTracking.chamberResolutionY=0.4;MCHTracking.sigmaCutForTracking=7.;MCHTracking.sigmaCutForImprovement=6.;"
    fi
    has_detector_reco ITS && [[ $RUNTYPE != "COSMICS" ]] && CONFIG_EXTRA_PROCESS_o2_mch_reco_workflow+="MCHTimeClusterizer.irFramesOnly=true;"
    [[ ! -z ${CUT_RANDOM_FRACTION_MCH:-} ]] && CONFIG_EXTRA_PROCESS_o2_mch_reco_workflow+="MCHTimeClusterizer.rofRejectionFraction=$CUT_RANDOM_FRACTION_MCH;"
  fi
  [[ $RUNTYPE == "COSMICS" ]] && [[ -z ${CONFIG_EXTRA_PROCESS_o2_mft_reco_workflow:-} ]] && CONFIG_EXTRA_PROCESS_o2_mft_reco_workflow="MFTTracking.FullClusterScan=true"
fi
[[ "${ED_VERTEX_MODE:-}" == "1" ]] && has_detectors_reco ITS && has_detector_matching PRIMVTX && [[ ! -z "$VERTEXING_SOURCES" ]] && EVE_CONFIG+=" --primary-vertex-mode"
[[ $SYNCRAWMODE == 1 ]] && [[ -z ${CONFIG_EXTRA_PROCESS_o2_trd_global_tracking:-} ]] && CONFIG_EXTRA_PROCESS_o2_trd_global_tracking='GPU_rec_trd.maxChi2=25;GPU_rec_trd.penaltyChi2=20;GPU_rec_trd.extraRoadY=4;GPU_rec_trd.extraRoadZ=10;GPU_rec_trd.applyDeflectionCut=0;GPU_rec_trd.trkltResRPhiIdeal=1'
[[ $SYNCRAWMODE == 1 ]] && [[ -z ${ARGS_EXTRA_PROCESS_o2_phos_reco_workflow:-} ]] && ARGS_EXTRA_PROCESS_o2_phos_reco_workflow='--presamples 2 --fitmethod semigaus'

# ---------------------------------------------------------------------------------------------------------------------
# Start of workflow command generation

WORKFLOW= # Make sure we start with an empty workflow
[[ "${GEN_TOPO_ONTHEFLY:-}" == "1" ]] && WORKFLOW="echo '{}' | "

# ---------------------------------------------------------------------------------------------------------------------
# Input workflow
INPUT_DETECTOR_LIST=$WORKFLOW_DETECTORS
if [[ ! -z ${WORKFLOW_DETECTORS_USE_GLOBAL_READER} ]]; then
  for i in ${WORKFLOW_DETECTORS_USE_GLOBAL_READER//,/ }; do
    export INPUT_DETECTOR_LIST=$(echo $INPUT_DETECTOR_LIST | sed -e "s/,$i,/,/g" -e "s/^$i,//" -e "s/,$i"'$'"//" -e "s/^$i"'$'"//")
  done
  GLOBAL_READER_OPTIONS=
  has_detector ITS && SYNCMODE==1 && GLOBAL_READER_OPTIONS+=" --ir-frames-its"
  add_W o2-global-track-cluster-reader "--cluster-types $WORKFLOW_DETECTORS_USE_GLOBAL_READER --track-types $WORKFLOW_DETECTORS_USE_GLOBAL_READER $GLOBAL_READER_OPTIONS $DISABLE_MC --hbfutils-config o2_tfidinfo.root"
  has_detector FV0 && has_detector_from_global_reader FV0 && add_W o2-fv0-digit-reader-workflow "$DISABLE_MC --hbfutils-config o2_tfidinfo.root --fv0-digit-infile o2_fv0digits.root"
  has_detector MID && has_detector_from_global_reader MID && add_W o2-mid-digits-reader-workflow "$DISABLE_MC --hbfutils-config o2_tfidinfo.root --mid-digit-infile mid-digits-decoded.root"
  has_detector MCH && has_detector_from_global_reader MCH && add_W o2-mch-digits-reader-workflow "$DISABLE_MC --hbfutils-config o2_tfidinfo.root --infile mchdigits.root"
  has_detector MCH && has_detector_from_global_reader MCH && add_W o2-mch-clusters-reader-workflow "--hbfutils-config o2_tfidinfo.root" "" 0
  has_detector MCH && has_detector_from_global_reader MCH && add_W o2-mch-preclusters-reader-workflow "--hbfutils-config o2_tfidinfo.root" "" 0
  has_detector TRD && has_detector_from_global_reader TRD && add_W o2-trd-digit-reader-workflow "--hbfutils-config o2_tfidinfo.root"
  has_detector TOF && has_detector_from_global_reader TOF && add_W o2-tof-reco-workflow "$DISABLE_MC --input-type digits --output-type NONE --hbfutils-config o2_tfidinfo.root"
fi

if [[ ! -z $INPUT_DETECTOR_LIST ]]; then
  if [[ $CTFINPUT == 1 ]]; then
    GPU_INPUT=compressed-clusters-ctf
    TOF_INPUT=digits
    CTFName=`ls -t $RAWINPUTDIR/o2_ctf_*.root 2> /dev/null | head -n1`
    [[ -z $CTFName && $WORKFLOWMODE == "print" ]] && CTFName='$CTFName'
    [[ ! -z ${INPUT_FILE_LIST:-} ]] && CTFName=$INPUT_FILE_LIST
    if [[ -z $CTFName && $WORKFLOWMODE != "print" ]]; then echo "No CTF file given!"; exit 1; fi
    if [[ $NTIMEFRAMES == -1 ]]; then NTIMEFRAMES_CMD= ; else NTIMEFRAMES_CMD="--max-tf $NTIMEFRAMES"; fi
    CTF_EMC_SUBSPEC=
    ( workflow_has_parameter AOD || [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-emcal-cell-writer-workflow ) && has_detector EMC && CTF_EMC_SUBSPEC="--emcal-decoded-subspec 1"
    add_W o2-ctf-reader-workflow "--delay $TFDELAY --loop $TFLOOP $NTIMEFRAMES_CMD --ctf-input ${CTFName} ${INPUT_FILE_COPY_CMD+--copy-cmd} ${INPUT_FILE_COPY_CMD:-} --onlyDet $INPUT_DETECTOR_LIST $CTF_EMC_SUBSPEC ${TIMEFRAME_SHM_LIMIT+--timeframes-shm-limit} ${TIMEFRAME_SHM_LIMIT:-} --pipeline $(get_N tpc-entropy-decoder TPC REST 1 TPCENTDEC)"
  elif [[ $RAWTFINPUT == 1 ]]; then
    TFName=`ls -t $RAWINPUTDIR/o2_*.tf 2> /dev/null | head -n1`
    [[ -z $TFName && $WORKFLOWMODE == "print" ]] && TFName='$TFName'
    [[ ! -z ${INPUT_FILE_LIST:-} ]] && TFName=$INPUT_FILE_LIST
    if [[ -z $TFName && $WORKFLOWMODE != "print" ]]; then echo "No raw file given!"; exit 1; fi
    if [[ $NTIMEFRAMES == -1 ]]; then NTIMEFRAMES_CMD= ; else NTIMEFRAMES_CMD="--max-tf $NTIMEFRAMES"; fi
    if [[ -z $WORKFLOW_DETECTORS_FLP_PROCESSING || $WORKFLOW_DETECTORS_FLP_PROCESSING == "NONE" ]]; then
      TFRAWOPT="--raw-only-det all"
    else
      TFRAWOPT="--non-raw-only-det $WORKFLOW_DETECTORS_FLP_PROCESSING"
    fi
    add_W o2-raw-tf-reader-workflow "--delay $TFDELAY $TFRAWOPT --loop $TFLOOP $NTIMEFRAMES_CMD --input-data ${TFName} ${INPUT_FILE_COPY_CMD+--copy-cmd} ${INPUT_FILE_COPY_CMD:-} --onlyDet $INPUT_DETECTOR_LIST ${TIMEFRAME_SHM_LIMIT+--timeframes-shm-limit} ${TIMEFRAME_SHM_LIMIT:-}"
  elif [[ $EXTINPUT == 1 ]]; then
    PROXY_CHANNEL="name=readout-proxy,type=pull,method=connect,address=ipc://${UDS_PREFIX}${INRAWCHANNAME},transport=shmem,rateLogging=$EPNSYNCMODE"
    PROXY_INSPEC="dd:FLP/DISTSUBTIMEFRAME/0"
    PROXY_IN_N=0
    for i in ${INPUT_DETECTOR_LIST//,/ }; do
      if has_detector_flp_processing $i; then
        case $i in
          TOF)
            PROXY_INTYPE="CRAWDATA";;
          FT0 | FV0 | FDD)
            PROXY_INTYPE="DIGITSBC/0 DIGITSCH/0";;
          PHS)
            PROXY_INTYPE="CELLS CELLTRIGREC";;
          CPV)
            PROXY_INTYPE="DIGITS/0 DIGITTRIGREC/0 RAWHWERRORS";;
          EMC)
            PROXY_INTYPE="CELLS/0 CELLSTRGR/0 DECODERERR";;
          CTP)
            PROXY_INTYPE="LUMI/0 RAWDATA"
            CTP_CONFIG=" --no-lumi "
            ;;
          *)
            echo Input type for detector $i with FLP processing not defined 1>&2
            exit 1;;
        esac
      else
        PROXY_INTYPE=RAWDATA
      fi
      for j in $PROXY_INTYPE; do
        PROXY_INNAME="RAWIN$PROXY_IN_N"
        let PROXY_IN_N=$PROXY_IN_N+1
        PROXY_INSPEC+=";$PROXY_INNAME:$i/$j"
      done
    done
    [[ ! -z ${TIMEFRAME_RATE_LIMIT:-} ]] && [[ $TIMEFRAME_RATE_LIMIT != 0 ]] && PROXY_CHANNEL+=";name=metric-feedback,type=pull,method=connect,address=ipc://${UDS_PREFIX}metric-feedback-${O2JOBID:-$NUMAID},transport=shmem,rateLogging=0"
    add_W o2-dpl-raw-proxy "--dataspec \"$PROXY_INSPEC\" --readout-proxy \"--channel-config \\\"$PROXY_CHANNEL\\\"\" ${TIMEFRAME_SHM_LIMIT+--timeframes-shm-limit} ${TIMEFRAME_SHM_LIMIT:-}" "" 0
  elif [[ $DIGITINPUT == 1 ]]; then
    [[ $NTIMEFRAMES != 1 ]] && { echo "Digit input works only with NTIMEFRAMES=1" 1>&2; exit 1; }
    DISABLE_DIGIT_ROOT_INPUT=
    DISABLE_DIGIT_CLUSTER_INPUT=
    TOF_INPUT=digits
    GPU_INPUT=zsonthefly
    has_detector TPC && add_W o2-tpc-reco-workflow "--input-type digits --output-type zsraw,disable-writer $DISABLE_MC --pipeline $(get_N tpc-zsEncoder TPC RAW 1 TPCRAWDEC)"
    has_detector MID && add_W o2-mid-digits-reader-workflow "$DISABLE_MC" ""
  else
    if [[ $NTIMEFRAMES == -1 ]]; then NTIMEFRAMES_CMD= ; else NTIMEFRAMES_CMD="--max-tf 0 --loop $NTIMEFRAMES"; fi
    [[ ! -f $RAWINPUTDIR/rawAll.cfg ]] && { echo "rawAll.cfg missing" 1>&2; exit 1; }
    add_W o2-raw-file-reader-workflow "--detect-tf0 --delay $TFDELAY $NTIMEFRAMES_CMD --input-conf $RAWINPUTDIR/rawAll.cfg --onlyDet $INPUT_DETECTOR_LIST ${TIMEFRAME_SHM_LIMIT+--timeframes-shm-limit} ${TIMEFRAME_SHM_LIMIT:-}" "HBFUtils.nHBFPerTF=$NHBPERTF"
  fi
fi

# if root output is requested, record info of processed TFs DataHeader for replay of root files
ROOT_OUTPUT_ASKED=`declare -p | cut -d' ' -f3 | cut -d'=' -f1 | grep ENABLE_ROOT_OUTPUT_`
[[ -z "$DISABLE_ROOT_OUTPUT" ]] || [[ ! -z $ROOT_OUTPUT_ASKED ]] && add_W o2-tfidinfo-writer-workflow

# ---------------------------------------------------------------------------------------------------------------------
# Raw decoder workflows - disabled in async mode
if [[ $CTFINPUT == 0 && $DIGITINPUT == 0 ]]; then
  if has_detector TPC && [[ "${TPC_CONVERT_LINKZS_TO_RAW:-}" == "1" ]]; then
    GPU_INPUT=zsonthefly
    add_W o2-tpc-raw-to-digits-workflow "--input-spec \"\" --remove-duplicates --pipeline $(get_N tpc-raw-to-digits-0 TPC RAW 1 TPCRAWDEC)"
    add_W o2-tpc-reco-workflow "--input-type digitizer --output-type zsraw,disable-writer --pipeline $(get_N tpc-zsEncoder TPC RAW 1 TPCRAWDEC)"
  fi
  has_detector ITS && ! has_detector_from_global_reader ITS && add_W o2-itsmft-stf-decoder-workflow "--nthreads ${NITSDECTHREADS} --raw-data-dumps $ALPIDE_ERR_DUMPS --pipeline $(get_N its-stf-decoder ITS RAW 1 ITSRAWDEC)" "$ITSMFT_STROBES;VerbosityConfig.rawParserSeverity=warn;"
  has_detector MFT && ! has_detector_from_global_reader MFT && add_W o2-itsmft-stf-decoder-workflow "--nthreads ${NMFTDECTHREADS} --raw-data-dumps $ALPIDE_ERR_DUMPS --pipeline $(get_N mft-stf-decoder MFT RAW 1 MFTRAWDEC) --runmft true" "$ITSMFT_STROBES;VerbosityConfig.rawParserSeverity=warn;"
  has_detector FT0 && ! has_detector_from_global_reader FT0 && ! has_detector_flp_processing FT0 && add_W o2-ft0-flp-dpl-workflow "$DISABLE_ROOT_OUTPUT --pipeline $(get_N ft0-datareader-dpl FT0 RAW 1)"
  has_detector FV0 && ! has_detector_from_global_reader FV0 && ! has_detector_flp_processing FV0 && add_W o2-fv0-flp-dpl-workflow "$DISABLE_ROOT_OUTPUT --pipeline $(get_N fv0-datareader-dpl FV0 RAW 1)"
  has_detector MID && ! has_detector_from_global_reader MID && add_W o2-mid-raw-to-digits-workflow "$MIDDEC_CONFIG --pipeline $(get_N MIDRawDecoder MID RAW 1),$(get_N MIDDecodedDataAggregator MID RAW 1)"
  has_detector MCH && ! has_detector_from_global_reader MCH && add_W o2-mch-raw-to-digits-workflow "--pipeline $(get_N mch-data-decoder MCH RAW 1)"
  has_detector TOF && ! has_detector_from_global_reader TOF && ! has_detector_flp_processing TOF && add_W o2-tof-compressor "--pipeline $(get_N tof-compressor-0 TOF RAW 1)"
  has_detector FDD && ! has_detector_from_global_reader FDD && ! has_detector_flp_processing FDD && add_W o2-fdd-flp-dpl-workflow "$DISABLE_ROOT_OUTPUT --pipeline $(get_N fdd-datareader-dpl FDD RAW 1)"
  has_detector TRD && ! has_detector_from_global_reader TRD && add_W o2-trd-datareader "$DISABLE_ROOT_OUTPUT --pipeline $(get_N trd-datareader TRD RAW 1 TRDRAWDEC)" "" 0
  has_detector ZDC && ! has_detector_from_global_reader ZDC && add_W o2-zdc-raw2digits "$DISABLE_ROOT_OUTPUT --pipeline $(get_N zdc-datareader-dpl ZDC RAW 1)"
  has_detector HMP && ! has_detector_from_global_reader HMP && add_W o2-hmpid-raw-to-digits-stream-workflow "--pipeline $(get_N HMP-RawStreamDecoder HMP RAW 1)"
  has_detector CTP && ! has_detector_from_global_reader CTP && add_W o2-ctp-reco-workflow "$DISABLE_ROOT_OUTPUT $CTP_CONFIG --ntf-to-average 1 --pipeline $(get_N ctp-raw-decoder CTP RAW 1)"
  has_detector PHS && ! has_detector_from_global_reader PHS && ! has_detector_flp_processing PHS && add_W o2-phos-reco-workflow "--input-type raw --output-type cells $DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT --pipeline $(get_N PHOSRawToCellConverterSpec PHS REST 1) $DISABLE_MC"
  has_detector CPV && ! has_detector_from_global_reader CPV && add_W o2-cpv-reco-workflow "--input-type $CPV_INPUT --output-type clusters $DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT --pipeline $(get_N CPVRawToDigitConverterSpec CPV REST 1),$(get_N CPVClusterizerSpec CPV REST 1) $DISABLE_MC"
  has_detector EMC && ! has_detector_from_global_reader EMC && ! has_detector_flp_processing EMC && add_W o2-emcal-reco-workflow "--input-type raw --output-type cells ${RAW_EMC_SUBSPEC:-} $EMCRAW2C_CONFIG $DISABLE_ROOT_OUTPUT $DISABLE_MC --pipeline $(get_N EMCALRawToCellConverterSpec EMC REST 1 EMCREC)"
fi

# ---------------------------------------------------------------------------------------------------------------------
# Common reconstruction workflows
(has_detector_reco TPC || has_detector_ctf TPC) && ! has_detector_from_global_reader TPC && add_W o2-gpu-reco-workflow "--gpu-reconstruction \"$GPU_CONFIG_SELF\" --input-type=$GPU_INPUT $DISABLE_MC --output-type $GPU_OUTPUT --pipeline gpu-reconstruction:${N_TPCTRK:-1} $GPU_CONFIG" "GPU_global.deviceType=$GPUTYPE;GPU_proc.debugLevel=0;$GPU_CONFIG_KEY;$TRACKTUNETPCINNER"
(has_detector_reco TOF || has_detector_ctf TOF) && ! has_detector_from_global_reader TOF && add_W o2-tof-reco-workflow "$TOF_CONFIG --input-type $TOF_INPUT --output-type $TOF_OUTPUT $DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC --pipeline $(get_N tof-compressed-decoder TOF RAW 1),$(get_N TOFClusterer TOF REST 1)"
has_detector_reco ITS && ! has_detector_from_global_reader ITS && add_W o2-its-reco-workflow "--trackerCA $ITS_CONFIG $DISABLE_MC $DISABLE_DIGIT_CLUSTER_INPUT $DISABLE_ROOT_OUTPUT --pipeline $(get_N its-tracker ITS REST 1 ITSTRK)" "$ITS_CONFIG_KEY;$ITSMFT_STROBES;$ITSEXTRAERR"
has_detector_reco FT0 && ! has_detector_from_global_reader FT0 && add_W o2-ft0-reco-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC --pipeline $(get_N ft0-reconstructor FT0 REST 1)"
has_detector_reco TRD && ! has_detector_from_global_reader TRD && add_W o2-trd-tracklet-transformer "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC $TRD_FILTER_CONFIG --pipeline $(get_N TRDTRACKLETTRANSFORMER TRD REST 1 TRDTRKTRANS)"
has_detectors_reco ITS TPC && has_detector_matching ITSTPC && add_W o2-tpcits-match-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC $SEND_ITSTPC_DTGL  $TPC_CORR_SCALING --nthreads $ITSTPC_THREADS --pipeline $(get_N itstpc-track-matcher MATCH REST $ITSTPC_THREADS TPCITS)" "$ITSTPC_CONFIG_KEY;$ITSMFT_STROBES;$ITSEXTRAERR"
has_detector_reco TRD && [[ ! -z "$TRD_SOURCES" ]] && add_W o2-trd-global-tracking "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC $TRD_CONFIG $TRD_FILTER_CONFIG $TPC_CORR_SCALING --track-sources $TRD_SOURCES --pipeline $(get_N trd-globaltracking_TPC_ITS-TPC_ TRD REST 1 TRDTRK),$(get_N trd-globaltracking_TPC_FT0_ITS-TPC_ TRD REST 1 TRDTRK)" "$TRD_CONFIG_KEY;$ITSMFT_STROBES;$ITSEXTRAERR"
has_detector_reco TOF && [[ ! -z "$TOF_SOURCES" ]] && add_W o2-tof-matcher-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC $TPC_CORR_SCALING --track-sources $TOF_SOURCES --pipeline $(get_N tof-matcher TOF REST 1 TOFMATCH)" "$ITSMFT_STROBES;$ITSEXTRAERR"
has_detectors TPC && [[ -z "$DISABLE_ROOT_OUTPUT" && "${SKIP_TPC_CLUSTERSTRACKS_OUTPUT:-}" != 1 ]] && add_W o2-tpc-reco-workflow "--input-type pass-through --output-type clusters,tracks,send-clusters-per-sector $DISABLE_MC"

# ---------------------------------------------------------------------------------------------------------------------
# Reconstruction workflows normally active only in async mode in async mode ($LIST_OF_ASYNC_RECO_STEPS), but can be forced via $WORKFLOW_EXTRA_PROCESSING_STEPS
has_detector MID && ! has_detector_from_global_reader MID && has_processing_step MID_RECO && add_W o2-mid-reco-workflow "$DISABLE_ROOT_OUTPUT $DISABLE_MC --pipeline $(get_N MIDClusterizer MID REST 1),$(get_N MIDTracker MID REST 1)"
has_detector MCH && ! has_detector_from_global_reader MCH && has_processing_step MCH_RECO && add_W o2-mch-reco-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC --pipeline $(get_N mch-track-finder MCH REST 1 MCHTRK),$(get_N mch-cluster-finder MCH REST 1 MCHCL),$(get_N mch-cluster-transformer MCH REST 1)" "$MCH_CONFIG_KEY"
has_detector MFT && ! has_detector_from_global_reader MFT && has_processing_step MFT_RECO && add_W o2-mft-reco-workflow "$DISABLE_DIGIT_CLUSTER_INPUT $DISABLE_MC $DISABLE_ROOT_OUTPUT $MFT_CONFIG --pipeline $(get_N mft-tracker MFT REST 1 MFTTRK)" "$MFT_CONFIG_KEY;$ITSMFT_STROBES"
has_detector FDD && ! has_detector_from_global_reader FDD && has_processing_step FDD_RECO && add_W o2-fdd-reco-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC"
has_detector FV0 && ! has_detector_from_global_reader FV0 && has_processing_step FV0_RECO && add_W o2-fv0-reco-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC"
has_detector ZDC && ! has_detector_from_global_reader ZDC && has_processing_step ZDC_RECO && add_W o2-zdc-digits-reco "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC"
has_detector HMP && ! has_detector_from_global_reader HMP && has_processing_step HMP_RECO && add_W o2-hmpid-digits-to-clusters-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT --pipeline $(get_N HMP-Clusterization HMP REST 1 HMPCLUS)"
has_detectors_reco MCH MID && has_detector_matching MCHMID && add_W o2-muon-tracks-matcher-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_MC $DISABLE_ROOT_OUTPUT --pipeline $(get_N muon-track-matcher MATCH REST 1)"
has_detectors_reco MFT MCH && has_detector_matching MFTMCH && add_W o2-globalfwd-matcher-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC --pipeline $(get_N globalfwd-track-matcher MATCH REST 1 FWDMATCH)" "$MFTMCHConf"

# ---------------------------------------------------------------------------------------------------------------------
# Reconstruction workflows needed only in case QC or CALIB was requested
( has_detector_qc PHS || has_detector_calib PHS ) && ( workflow_has_parameter QC || workflow_has_parameter CALIB ) && add_W o2-phos-reco-workflow "--input-type cells --output-type clusters ${PHS_CONFIG} $DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $DISABLE_MC --pipeline $(get_N PHOSClusterizerSpec PHS REST 1)"

# ---------------------------------------------------------------------------------------------------------------------
# Reconstruction workflows applying detector-specific calibrations
( workflow_has_parameter AOD || [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-emcal-cell-writer-workflow ) && ! has_detector_from_global_reader EMC && has_detector EMC && add_W o2-emcal-cell-recalibrator-workflow "--input-subspec 1 --output-subspec 0 --redirect-led"

# ---------------------------------------------------------------------------------------------------------------------
# Writers for detectors whose reco workflow cannot write the output
has_detector_reco TRD && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-trd-digittracklet-writer ) && ! has_detector_from_global_reader TRD && add_W o2-trd-digittracklet-writer
has_detector_reco TPC && [[ "0${TPC_CONVERT_LINKZS_TO_RAW:-}" == "01" ]] && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-gpu-reco-workflow ) && ! has_detector_from_global_reader TPC && add_W o2-tpc-reco-workflow "--input-type digitizer --output-type digits $DISABLE_MC"
has_detector_reco CPV && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-cpv-cluster-writer-workflow ) && ! has_detector_from_global_reader CPV && add_W o2-cpv-cluster-writer-workflow "$DISABLE_MC"
( workflow_has_parameter AOD || [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-emcal-cell-writer-workflow ) && ! has_detector_from_global_reader EMC && has_detector EMC && add_W o2-emcal-cell-writer-workflow "--disable-mc --subspec 10 --cell-writer-name emcal-led-cells-writer --emcal-led-cells-writer \"--outfile emcledcells.root\""
[[ $CTFINPUT == 1 ]] && has_detector CTP && ! has_detector_from_global_reader CTP && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-ctp-digit-writer ) && add_W o2-ctp-digit-writer "$DISABLE_ROOT_OUTPUT"
has_detector_reco EMC && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-emcal-cell-writer-workflow ) && ! has_detector_from_global_reader EMC && add_W o2-emcal-cell-writer-workflow "$DISABLE_MC"
has_detector_reco PHS && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-phos-cell-writer-workflow ) && ! has_detector_from_global_reader PHS && add_W o2-phos-cell-writer-workflow "$DISABLE_MC"
has_detector_reco FV0 && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-fv0-digits-writer-workflow ) && ! has_detector_from_global_reader FV0 && add_W o2-fv0-digits-writer-workflow "$DISABLE_MC"
has_detector_reco MID && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-mid-decoded-digits-writer-workflow ) && ! has_detector_from_global_reader MID && add_W o2-mid-decoded-digits-writer-workflow "--mid-digits-tree-name o2sim" "" 0
has_detector_reco MCH && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-mch-digits-writer-workflow ) && ! has_detector_from_global_reader MCH && add_W o2-mch-digits-writer-workflow "" "" 0
has_detector_reco MCH && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-mch-clusters-writer-workflow ) && ! has_detector_from_global_reader MCH && add_W o2-mch-clusters-writer-workflow "" "" 0
has_detector_reco MCH && ( [[ -z "$DISABLE_ROOT_OUTPUT" ]] || needs_root_output o2-mch-preclusters-writer-workflow ) && ! has_detector_from_global_reader MCH && add_W o2-mch-preclusters-writer-workflow "" "" 0

# always run vertexing if requested and if there are some sources, but in cosmic mode we work in pass-trough mode (create record for non-associated tracks)
( [[ $BEAMTYPE == "cosmic" ]] || ! has_detector_reco ITS) && PVERTEX_CONFIG+=" --skip"
has_detector_matching PRIMVTX && [[ ! -z "$VERTEXING_SOURCES" ]] && add_W o2-primary-vertexing-workflow "$DISABLE_MC $DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $PVERTEX_CONFIG --pipeline $(get_N primary-vertexing MATCH REST 1 PRIMVTX),$(get_N pvertex-track-matching MATCH REST 1 PRIMVTXMATCH)" "${PVERTEXING_CONFIG_KEY}"

if [[ $BEAMTYPE != "cosmic" ]]; then
  has_detectors_reco ITS && has_detector_matching SECVTX && [[ ! -z "$SVERTEXING_SOURCES" ]] && add_W o2-secondary-vertexing-workflow "$DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT $TPC_CORR_SCALING --vertexing-sources $SVERTEXING_SOURCES --threads $SVERTEX_THREADS --pipeline $(get_N secondary-vertexing MATCH REST $SVERTEX_THREADS SECVTX)"
  has_detectors_reco ITS && has_detector_matching SECVTX && has_detector_matching STRK && add_W o2-strangeness-tracking-workflow "$DISABLE_MC $DISABLE_DIGIT_ROOT_INPUT $DISABLE_ROOT_OUTPUT --pipeline $(get_N strangeness-tracking MATCH REST 1 STRK)"
fi

# ---------------------------------------------------------------------------------------------------------------------
# Entropy encoding / ctf creation workflows - disabled in async mode
if has_processing_step ENTROPY_ENCODER && [[ ! -z "$WORKFLOW_DETECTORS_CTF" ]] && [[ $WORKFLOW_DETECTORS_CTF != "NONE" ]]; then
  # Entropy encoder workflows
  has_detector_ctf MFT && add_W o2-itsmft-entropy-encoder-workflow "--mem-factor ${MFT_ENC_MEMFACT:-1.5} --runmft true --pipeline $(get_N mft-entropy-encoder MFT CTF 1)"
  has_detector_ctf FT0 && add_W o2-ft0-entropy-encoder-workflow "--mem-factor ${FT0_ENC_MEMFACT:-1.5} --pipeline $(get_N ft0-entropy-encoder FT0 CTF 1)"
  has_detector_ctf FV0 && add_W o2-fv0-entropy-encoder-workflow "--mem-factor ${FV0_ENC_MEMFACT:-1.5} --pipeline $(get_N fv0-entropy-encoder FV0 CTF 1)"
  has_detector_ctf MID && add_W o2-mid-entropy-encoder-workflow "--mem-factor ${MID_ENC_MEMFACT:-1.5} --pipeline $(get_N mid-entropy-encoder MID CTF 1)"
  has_detector_ctf MCH && add_W o2-mch-entropy-encoder-workflow "--mem-factor ${MCH_ENC_MEMFACT:-1.5} --pipeline $(get_N mch-entropy-encoder MCH CTF 1)"
  has_detector_ctf PHS && add_W o2-phos-entropy-encoder-workflow "--mem-factor ${PHS_ENC_MEMFACT:-1.5} --pipeline $(get_N phos-entropy-encoder PHS CTF 1)"
  has_detector_ctf CPV && add_W o2-cpv-entropy-encoder-workflow "--mem-factor ${CPV_ENC_MEMFACT:-1.5} --pipeline $(get_N cpv-entropy-encoder CPV CTF 1)"
  has_detector_ctf EMC && add_W o2-emcal-entropy-encoder-workflow "--mem-factor ${EMC_ENC_MEMFACT:-1.5} --pipeline $(get_N emcal-entropy-encoder EMC CTF 1)"
  has_detector_ctf ZDC && add_W o2-zdc-entropy-encoder-workflow "--mem-factor ${ZDC_ENC_MEMFACT:-1.5} --pipeline $(get_N zdc-entropy-encoder ZDC CTF 1)"
  has_detector_ctf FDD && add_W o2-fdd-entropy-encoder-workflow "--mem-factor ${FDD_ENC_MEMFACT:-1.5} --pipeline $(get_N fdd-entropy-encoder FDD CTF 1)"
  has_detector_ctf HMP && add_W o2-hmpid-entropy-encoder-workflow "--mem-factor ${HMP_ENC_MEMFACT:-1.5} --pipeline $(get_N hmpid-entropy-encoder HMP CTF 1)"
  has_detector_ctf TOF && add_W o2-tof-entropy-encoder-workflow "--mem-factor ${TOF_ENC_MEMFACT:-1.5} --pipeline $(get_N tof-entropy-encoder TOF CTF 1)"
  has_detector_ctf ITS && add_W o2-itsmft-entropy-encoder-workflow "--mem-factor ${ITS_ENC_MEMFACT:-1.5} --pipeline $(get_N its-entropy-encoder ITS CTF 1)"
  has_detector_ctf TRD && add_W o2-trd-entropy-encoder-workflow "--mem-factor ${TRD_ENC_MEMFACT:-1.5} --pipeline $(get_N trd-entropy-encoder TRD CTF 1 TRDENT)"
  has_detector_ctf TPC && add_W o2-tpc-reco-workflow "--mem-factor ${TPC_ENC_MEMFACT:-1.} --input-type compressed-clusters-flat --output-type encoded-clusters,disable-writer --pipeline $(get_N tpc-entropy-encoder TPC CTF 1 TPCENT)"
  has_detector_ctf CTP && add_W o2-ctp-entropy-encoder-workflow "--mem-factor ${CTP_ENC_MEMFACT:-1.5} --pipeline $(get_N its-entropy-encoder CTP CTF 1)"

  if [[ $CREATECTFDICT == 1 && $WORKFLOWMODE == "run" ]] ; then
    [[ -f $CTF_DICT ]] && rm -f $CTF_DICT
  fi
  CTF_OUTPUT_TYPE="none"
  if [[ $CREATECTFDICT == 1 ]] && [[ $SAVECTF == 1 ]]; then CTF_OUTPUT_TYPE="both"; fi
  if [[ $CREATECTFDICT == 1 ]] && [[ $SAVECTF == 0 ]]; then CTF_OUTPUT_TYPE="dict"; fi
  if [[ $CREATECTFDICT == 0 ]] && [[ $SAVECTF == 1 ]]; then CTF_OUTPUT_TYPE="ctf"; fi
  if [[ $EPNSYNCMODE == 1 ]]; then
    CTF_CONFIG="--report-data-size-interval 1000"
  else
    CTF_CONFIG="--report-data-size-interval 1"
  fi
  CONFIG_CTF="--output-dir \"$CTF_DIR\" $CTF_CONFIG --output-type $CTF_OUTPUT_TYPE --min-file-size ${CTF_MINSIZE} --max-ctf-per-file ${CTF_MAX_PER_FILE} --onlyDet ${WORKFLOW_DETECTORS_CTF/TST/} --meta-output-dir $EPN2EOS_METAFILES_DIR"
  if [[ $CREATECTFDICT == 1 ]] && [[ $EXTINPUT == 1 ]]; then CONFIG_CTF+=" --save-dict-after $SAVE_CTFDICT_NTIMEFRAMES"; fi
  [[ $EPNSYNCMODE == 1 ]] && CONFIG_CTF+=" --require-free-disk 53687091200 --wait-for-free-disk $CTF_FREE_DISK_WAIT --max-wait-for-free-disk $CTF_MAX_FREE_DISK_WAIT"
  add_W o2-ctf-writer-workflow "$CONFIG_CTF"
fi

# ---------------------------------------------------------------------------------------------------------------------
# Calibration workflows
workflow_has_parameter CALIB && { source ${CALIB_WF:-$GEN_TOPO_MYDIR/calib-workflow.sh}; [[ $? != 0 ]] && echo "calib-workflow.sh failed" 1>&2 && exit 1; }
workflow_has_parameters CALIB CALIB_LOCAL_INTEGRATED_AGGREGATOR && { source ${CALIB_AGGREGATOR_WF:-$GEN_TOPO_MYDIR/aggregator-workflow.sh}; [[ $? != 0 ]] && echo "aggregator-workflow.sh failed" 1>&2 && exit 1; }

# ---------------------------------------------------------------------------------------------------------------------
# Event display
# RS this is a temporary setting
: ${ED_TRACKS:=$TRACK_SOURCES}
: ${ED_CLUSTERS:=$TRACK_SOURCES}
workflow_has_parameter EVENT_DISPLAY && [[ $NUMAID == 0 ]] && [[ ! -z "$ED_TRACKS" ]] && [[ ! -z "$ED_CLUSTERS" ]] && add_W o2-eve-export-workflow "--display-tracks $ED_TRACKS --display-clusters $ED_CLUSTERS --skipOnEmptyInput $DISABLE_DIGIT_ROOT_INPUT --number-of_tracks 50000 $EVE_CONFIG $DISABLE_MC" "$ITSMFT_STROBES"

workflow_has_parameter GPU_DISPLAY && [[ $NUMAID == 0 ]] && add_W o2-gpu-display "${ED_TRACKS+--display-tracks} $ED_TRACKS ${ED_CLUSTERS+--display-clusters} $ED_CLUSTERS"

# ---------------------------------------------------------------------------------------------------------------------
# AOD
: ${AOD_INPUT:=$TRACK_SOURCES}
: ${AODPROD_OPT:=}
( ! has_detector_matching SECVTX || ! has_detectors_reco ITS || [[ $BEAMTYPE == "cosmic" ]]) && AODPROD_OPT+="--disable-secondary-vertices"
( ! has_detector_matching STRK || ! has_detector_matching SECVTX || ! has_detectors_reco ITS || [[ $BEAMTYPE == "cosmic" ]]) && AODPROD_OPT+="--disable-strangeness-tracking"

workflow_has_parameter AOD && [[ ! -z "$AOD_INPUT" ]] && add_W o2-aod-producer-workflow "$AODPROD_OPT --info-sources $AOD_INPUT $DISABLE_DIGIT_ROOT_INPUT --aod-writer-keep dangling --aod-writer-resfile \"AO2D\" --aod-writer-resmode UPDATE $DISABLE_MC --pipeline $(get_N aod-producer-workflow AOD REST 1 AODPROD)"

# ---------------------------------------------------------------------------------------------------------------------
# Quality Control
workflow_has_parameter QC && { source $O2DPG_ROOT/DATA/production/qc-workflow.sh; [[ $? != 0 ]] && echo "qc-workflow.sh failed" 1>&2 && exit 1; }

if [[ ! -z "${EXTRA_WORKFLOW:-}" ]]; then
  WORKFLOW+="$EXTRA_WORKFLOW"
fi

if [[ ! -z "${ADD_EXTRA_WORKFLOW:-}" ]]; then
  OLD_IFS=$IFS
  IFS=','
  for wf in $ADD_EXTRA_WORKFLOW; do
    [[ ! -z "$wf" ]] && add_W $wf
  done
  IFS="$OLD_IFS"
fi

# ---------------------------------------------------------------------------------------------------------------------
# DPL run binary
WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT"

if [[ "${GEN_TOPO_AUTOSCALE_PROCESSES:-}" == "1" && (${GEN_TOPO_RUN_HOME_TEST:-} == 1 || $WORKFLOWMODE != "print") ]]; then
  TOTAL_N_PIPELINES=`echo "${WORKFLOW}" | grep -o ':\$((([0-9]*\*\$AUTOSCALE_PROCESS_FACTOR' | grep -o '[0-9]*' | awk '{s+=$1} END {print s}'`
  TOTAL_N_CPUCORES=$(($NUMAGPUIDS == 1 ? 64 : 128))
  if [[ -z $TOTAL_N_PIPELINES ]]; then
    AUTOSCALE_PROCESS_FACTOR=1
  else
    AUTOSCALE_PROCESS_FACTOR=$(($TOTAL_N_PIPELINES >= $TOTAL_N_CPUCORES || $TOTAL_N_PIPELINES == 0 ? 100 : ($TOTAL_N_CPUCORES * 100 / $TOTAL_N_PIPELINES)))
  fi
  [[ $WORKFLOWMODE == "print" || ${PRINT_WORKFLOW:-} == "1" ]] && echo "AUTOSCALE_PROCESS_FACTOR=$AUTOSCALE_PROCESS_FACTOR"
fi

# ---------------------------------------------------------------------------------------------------------------------
# Run / create / print workflow
if [[ "${FST_BENCHMARK_STARTUP:-}" == "1" ]]; then
  date 1>&2
  eval $WORKFLOW --dump > fst.startup.tmp.$NUMAID.json
  WORKFLOW2="cat fst.startup.tmp.$NUMAID.json | o2-dpl-run $ARGS_ALL $GLOBALDPLOPT"
  date 1>&2
  eval $WORKFLOW2
else
  [[ $WORKFLOWMODE != "print" ]] && WORKFLOW+=" --${WORKFLOWMODE} ${WORKFLOWMODE_FILE:-}"
  [[ $WORKFLOWMODE == "print" || ${PRINT_WORKFLOW:-} == "1" ]] && echo "#Workflow command:\n\n${WORKFLOW}\n" | sed -e "s/\\\\n/\n/g" -e"s/| */| \\\\\n/g" | eval cat $( [[ $WORKFLOWMODE == "dds" ]] && echo '1>&2')
  if [[ $WORKFLOWMODE != "print" ]]; then eval $WORKFLOW; else true; fi
fi

# ---------------------------------------------------------------------------------------------------------------------
