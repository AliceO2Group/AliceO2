#!/bin/bash

# ---------------------------------------------------------------------------------------------------------------------
# Get this script's directory and load common settings (use zsh first (e.g. on Mac) and fallback on `readlink -f` if zsh is not there)
command -v zsh > /dev/null 2>&1 && MYDIR=$(dirname $(zsh -c 'echo ${0:A}' "$0"))
test -z ${MYDIR+x} && MYDIR="$(dirname $(readlink -f $0))"
source $MYDIR/setenv.sh

# ---------------------------------------------------------------------------------------------------------------------
#Some additional settings used in this workflow
if [ -z $OPTIMIZED_PARALLEL_ASYNC ]; then OPTIMIZED_PARALLEL_ASYNC=0; fi  # Enable tuned process multiplicities for async processing on the EPN
if [ -z $CTF_DIR ];                  then CTF_DIR=$FILEWORKDIR; fi        # Directory where to store CTFs
if [ -z $CTF_DICT_DIR ];             then CTF_DICT_DIR=$FILEWORKDIR; fi   # Directory of CTF dictionaries
if [ -z $RECO_NUM_NODES_WORKFLOW ];  then RECO_NUM_NODES_WORKFLOW=250; fi # Number of EPNs running this workflow in parallel, to increase multiplicities if necessary, by default assume we are 1 out of 250 servers
if [ $SYNCMODE == 1 ]; then
  if [ -z "${WORKFLOW_DETECTORS_MATCHING+x}" ]; then export WORKFLOW_DETECTORS_MATCHING="ITSTPC,ITSTPCTRD,ITSTPCTOF"; fi # Select matchings that are enabled in sync mode
else
  if [ -z "${WORKFLOW_DETECTORS_MATCHING+x}" ]; then export WORKFLOW_DETECTORS_MATCHING="ALL"; fi # All matching / vertexing enabled in async mode
fi


workflow_has_parameter CTF && export SAVECTF=1
workflow_has_parameter GPU && { export GPUTYPE=HIP; export NGPUS=4; }

ITSCLUSDICT="${FILEWORKDIR}/ITSdictionary.bin"
MFTCLUSDICT="${FILEWORKDIR}/MFTdictionary.bin"
ITS_NOISE="${FILEWORKDIR}"
MFT_NOISE="${FILEWORKDIR}/mft_noise_220721_R3C-520.root"

MID_FEEID_MAP="$FILEWORKDIR/mid-feeId_mapper.txt"
CTF_MINSIZE="2000000"
NITSDECTHREADS=2
NMFTDECTHREADS=2
CTF_DICT=${CTF_DICT_DIR}/ctf_dictionary.root

ITSMFT_FILES="ITSClustererParam.dictFilePath=$ITSCLUSDICT;MFTClustererParam.dictFilePath=$MFTCLUSDICT";

if [ "0$O2_ROOT" == "0" ]; then
  eval "`alienv shell-helper`"
  alienv --no-refresh load O2/latest
fi

# ---------------------------------------------------------------------------------------------------------------------
# Set general arguments
ARGS_ALL="--session default --severity $SEVERITY --shm-segment-id $NUMAID --shm-segment-size $SHMSIZE $ARGS_ALL_EXTRA"
if [ $EPNMODE == 1 ]; then
  ARGS_ALL+=" --infologger-severity $INFOLOGGER_SEVERITY"
  ARGS_ALL+=" --monitoring-backend influxdb-unix:///tmp/telegraf.sock --resources-monitoring 60"
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
  ARGS_ALL+=" --shm-mlock-segment-on-creation 1"
fi
ARGS_ALL_CONFIG="NameConf.mDirGRP=$FILEWORKDIR;NameConf.mDirGeom=$FILEWORKDIR;NameConf.mDirCollContext=$FILEWORKDIR;NameConf.mDirMatLUT=$FILEWORKDIR;keyval.input_dir=$FILEWORKDIR;keyval.output_dir=/dev/null;$ALL_EXTRA_CONFIG"

# ---------------------------------------------------------------------------------------------------------------------
# Set some individual workflow arguments depending on configuration
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
CPV_INPUT=raw
EVE_CONFIG=" --jsons-folder $EDJSONS_DIR"
MIDDEC_CONFIG=

if [ $SYNCMODE == 1 ]; then
  if [ $BEAMTYPE == "PbPb" ]; then
    ITS_CONFIG_KEY+="fastMultConfig.cutMultClusLow=30;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500;"
  elif [ $BEAMTYPE == "pp" ]; then
    ITS_CONFIG_KEY+="fastMultConfig.cutMultClusLow=1;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500;"
  fi
  GPU_CONFIG_KEY+="GPU_global.synchronousProcessing=1;GPU_proc.clearO2OutputFromGPU=1;"
  TRD_CONFIG+=" --filter-trigrec"
  TRD_CONFIG_KEY+="GPU_proc.ompThreads=1;"
  TRD_TRANSFORMER_CONFIG+=" --filter-trigrec"
fi

has_detector_flp_processing CPV && CPV_INPUT=digits

if [ $CTFINPUT == 1 ]; then
  ITS_CONFIG+=" --tracking-mode async"
else
  ITS_CONFIG+=" --tracking-mode sync"
  GPU_OUTPUT+=",compressed-clusters-ctf"
fi

if [ $EPNMODE == 1 ]; then
  EVE_CONFIG+=" --eve-dds-collection-index 0"
  ITSMFT_FILES+=";ITSClustererParam.noiseFilePath=$ITS_NOISE;MFTClustererParam.noiseFilePath=$MFT_NOISE";
  MIDDEC_CONFIG+=" --feeId-config-file \"$MID_FEEID_MAP\""
  # Options for decoding current TRD real raw data (not needed for data converted from MC)
  if [ -z $TRD_DECODER_OPTIONS ]; then TRD_DECODER_OPTIONS=" --tracklethcheader 2  --ignore-digithcheader --halfchamberwords 2 --halfchambermajor 33 "; fi
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

if ! has_detector_reco TOF; then
  TOF_OUTPUT=digits
fi

# ---------------------------------------------------------------------------------------------------------------------
# Assemble matching sources
TRD_SOURCES=
TOF_SOURCES=
VTX_SOURCES=ITS
has_detector_matching ITSTPC && VTX_SOURCES+=",ITS-TPC"
has_detector_matching TPCTRD && (add_comma_separated TRD_SOURCES TPC; VTX_SOURCES+=",TPC-TRD")
has_detector_matching ITSTPCTRD && (add_comma_separated TRD_SOURCES ITS-TPC; VTX_SOURCES+=",ITS-TPC-TRD")
has_detector_matching TPCTOF && (add_comma_separated TOF_SOURCES TPC; VTX_SOURCES+=",TPC-TOF")
has_detector_matching ITSTPCTOF && (add_comma_separated TOF_SOURCES ITS-TPC; VTX_SOURCES+=",ITS-TPC-TOF")
has_detector_matching MFTMCH && VTX_SOURCES+=",MFT-MCH"
for det in `echo $LIST_OF_DETECTORS | sed "s/,/ /g" | grep -v "CTP"`; do
  has_detector_reco $det && VTX_SOURCES+=",$det"
done
PVERTEX_CONFIG="--vertexing-sources $VTX_SOURCES --vertex-track-matching-sources $VTX_SOURCES"
has_detector_reco FT0 && PVERTEX_CONFIG+=" --validate-with-ft0"
SECVERTEX_CONFIG="--vertexing-sources $VTX_SOURCES"

# ---------------------------------------------------------------------------------------------------------------------
# Process multiplicities
N_TPCTRK=1
N_TPCENT=1
N_TPCITS=1
N_ITSRAWDEC=1
N_MFTRAWDEC=1
N_CTPRAWDEC=1
N_TPCRAWDEC=$NGPUS
N_EMC=1
N_TRDENT=1
N_TRDTRK=1
N_TPCENTDEC=1
N_MFTTRK=1
N_ITSTRK=1
N_MCHTRK=1
N_TOFMATCH=1
N_F_REST=$MULTIPLICITY_FACTOR_REST
N_F_RAW=$MULTIPLICITY_FACTOR_RAWDECODERS
N_F_CTF=$MULTIPLICITY_FACTOR_CTFENCODERS
if [ $OPTIMIZED_PARALLEL_ASYNC != 0 ]; then
  # Tuned multiplicities for async Pb-Pb processing
  if [ $SYNCMODE == "1" ]; then echo "Must not use OPTIMIZED_PARALLEL_ASYNC with GPU or SYNCMODE" 1>&2; exit 1; fi
  if [ $NUMAGPUIDS == 1 ]; then N_NUMAFACTOR=1; else N_NUMAFACTOR=2; fi
  GPU_CONFIG_KEY+="GPU_proc.ompThreads=6;"
  TRD_CONFIG_KEY+="GPU_proc.ompThreads=2;"
  if [ $GPUTYPE == "CPU" ]; then
    N_TPCENTDEC=$((2 * $N_NUMAFACTOR))
    N_MFTTRK=$((3 * $N_NUMAFACTOR))
    N_ITSTRK=$((3 * $N_NUMAFACTOR))
    N_TPCITS=$((2 * $N_NUMAFACTOR))
    N_MCHTRK=$((1 * $N_NUMAFACTOR))
    N_TOFMATCH=$((9 * $N_NUMAFACTOR))
    N_TPCTRK=$((6 * $N_NUMAFACTOR))
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
  # Tuned multiplicities for sync Pb-Pb processing
  N_TPCENT=$((3 * $EPNPIPELINES * $NGPUS / 4 > 0 ? 3 * $EPNPIPELINES * $NGPUS / 4 : 1))
  N_TPCITS=$((3 * $EPNPIPELINES * $NGPUS / 4 > 0 ? 3 * $EPNPIPELINES * $NGPUS / 4 : 1))
  N_ITSRAWDEC=$((3 * $EPNPIPELINES * $NGPUS / 4 > 0 ? 3 * $EPNPIPELINES * $NGPUS / 4 : 1))
  N_EMC=$((7 * $EPNPIPELINES * $NGPUS / 4 > 0 ? 7 * $EPNPIPELINES * $NGPUS / 4 : 1))
  N_TRDENT=$((3 * $EPNPIPELINES * $NGPUS / 4 > 0 ? 3 * $EPNPIPELINES * $NGPUS / 4 : 1))
  N_TRDTRK=$((3 * $EPNPIPELINES * $NGPUS / 4 > 0 ? 3 * $EPNPIPELINES * $NGPUS / 4 : 1))
  N_TPCRAWDEC=$((6 * $EPNPIPELINES * $NGPUS / 4 > 0 ? 6 * $EPNPIPELINES * $NGPUS / 4 : 1))
  if [ $GPUTYPE == "CPU" ]; then
    N_TPCTRK=8
    GPU_CONFIG_KEY+="GPU_proc.ompThreads=4;"
  else
    N_TPCTRK=$NGPUS
  fi
fi
# Scale some multiplicities with the number of nodes
RECO_NUM_NODES_WORKFLOW_CMP=$(($RECO_NUM_NODES_WORKFLOW > 15 ? $RECO_NUM_NODES_WORKFLOW : 15)) # Limit the scaling factor
N_ITSRAWDEC=$((6 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP > $N_ITSRAWDEC ? 6 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP : $N_ITSRAWDEC))
N_MFTRAWDEC=$((6 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP > $N_MFTRAWDEC ? 6 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP : $N_MFTRAWDEC))
N_ITSTRK=$((2 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP > $N_ITSTRK ? 2 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP : $N_ITSTRK))
N_MFTTRK=$((2 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP > $N_MFTTRK ? 2 * 30 / $RECO_NUM_NODES_WORKFLOW_CMP : $N_MFTTRK))
N_CTPRAWDEC=$((30 / $RECO_NUM_NODES_WORKFLOW_CMP > N_CTPRAWDEC ? 30 / $RECO_NUM_NODES_WORKFLOW_CMP : $N_CTPRAWDEC))
# Apply external multiplicity factors
N_TPCTRK=$((N_TPCTRK * $N_F_REST))
N_TPCITS=$((N_TPCITS * $N_F_REST))
N_EMC=$((N_EMC * $N_F_REST))
N_TRDTRK=$((N_TRDTRK * $N_F_REST))
N_TPCENTDEC=$((N_TPCENTDEC * $N_F_REST))
N_MFTTRK=$((N_MFTTRK * $N_F_REST))
N_ITSTRK=$((N_ITSTRK * $N_F_REST))
N_MCHTRK=$((N_MCHTRK * $N_F_REST))
N_TOFMATCH=$((N_TOFMATCH * $N_F_REST))
N_TPCENT=$((N_TPCENT * $N_F_CTF))
N_TRDENT=$((N_TRDENT * $N_F_CTF))
N_ITSRAWDEC=$((N_ITSRAWDEC * $N_F_RAW))
N_MFTRAWDEC=$((N_MFTRAWDEC * $N_F_RAW))
N_TPCRAWDEC=$((N_TPCRAWDEC * $N_F_RAW))
N_CTPRAWDEC=$((N_CTPRAWDEC * $N_F_RAW))

# ---------------------------------------------------------------------------------------------------------------------
# Input workflow
if [ $CTFINPUT == 1 ]; then
  GPU_INPUT=compressed-clusters-ctf
  TOF_INPUT=digits
  CTFName=`ls -t $FILEWORKDIR/o2_ctf_*.root | head -n1`
  WORKFLOW="o2-ctf-reader-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --delay $TFDELAY --loop $NTIMEFRAMES --ctf-input ${CTFName} --ctf-dict ${CTF_DICT} --onlyDet $WORKFLOW_DETECTORS --pipeline tpc-entropy-decoder:$N_TPCENTDEC | "
elif [ $EXTINPUT == 1 ]; then
  PROXY_CHANNEL="name=readout-proxy,type=pull,method=connect,address=ipc://@$INRAWCHANNAME,transport=shmem,rateLogging=0"
  PROXY_INSPEC="dd:FLP/DISTSUBTIMEFRAME/0;eos:***/INFORMATION"
  PROXY_IN_N=0
  for i in `echo "$WORKFLOW_DETECTORS" | sed "s/,/ /g"`; do
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
  WORKFLOW="o2-dpl-raw-proxy $ARGS_ALL --dataspec \"$PROXY_INSPEC\" --channel-config \"$PROXY_CHANNEL\" | "
else
  WORKFLOW="o2-raw-file-reader-workflow --detect-tf0 $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG;HBFUtils.nHBFPerTF=$NHBPERTF;\" --delay $TFDELAY --loop $NTIMEFRAMES --max-tf 0 --input-conf $FILEWORKDIR/rawAll.cfg | "
fi

# ---------------------------------------------------------------------------------------------------------------------
# Raw decoder workflows - disabled in async mode
if [ $CTFINPUT == 0 ]; then
  if has_detector TPC && [ $EPNMODE == 1 ]; then
    GPU_INPUT=zsonthefly
    WORKFLOW+="o2-tpc-raw-to-digits-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-spec \"A:TPC/RAWDATA;dd:FLP/DISTSUBTIMEFRAME/0\" --remove-duplicates --pipeline tpc-raw-to-digits-0:$N_TPCRAWDEC | "
    WORKFLOW+="o2-tpc-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type digitizer --output-type zsraw,disable-writer --pipeline tpc-zsEncoder:$N_TPCRAWDEC | "
  fi
  has_detector ITS && WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --configKeyValues \"${ARGS_ALL_CONFIG};$ITSMFT_FILES\" --nthreads ${NITSDECTHREADS} --pipeline its-stf-decoder:$N_ITSRAWDEC | "
  has_detector MFT && WORKFLOW+="o2-itsmft-stf-decoder-workflow $ARGS_ALL --configKeyValues \"${ARGS_ALL_CONFIG};$ITSMFT_FILES\" --nthreads ${NMFTDECTHREADS} --pipeline mft-stf-decoder:$N_MFTRAWDEC --runmft true | "
  has_detector FT0 && ! has_detector_flp_processing FT0 && WORKFLOW+="o2-ft0-flp-dpl-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output --pipeline ft0-datareader-dpl:$N_F_RAW | "
  has_detector FV0 && ! has_detector_flp_processing FV0 && WORKFLOW+="o2-fv0-flp-dpl-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output --pipeline fv0-datareader-dpl:$N_F_RAW | "
  has_detector MID && WORKFLOW+="o2-mid-raw-to-digits-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" $MIDDEC_CONFIG --pipeline MIDRawDecoder:$N_F_RAW,MIDDecodedDataAggregator:$N_F_RAW | "
  has_detector MCH && WORKFLOW+="o2-mch-raw-to-digits-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline mch-data-decoder:$N_F_RAW | "
  has_detector TOF && ! has_detector_flp_processing TOF && WORKFLOW+="o2-tof-compressor $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" | "
  has_detector FDD && ! has_detector_flp_processing FDD && WORKFLOW+="o2-fdd-flp-dpl-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output --pipeline fdd-datareader-dpl:$N_F_RAW | "
  has_detector TRD && WORKFLOW+="o2-trd-datareader $ARGS_ALL $TRD_DECODER_OPTIONS --pipeline trd-datareader:$N_F_RAW | "
  has_detector ZDC && WORKFLOW+="o2-zdc-raw2digits $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output --pipeline zdc-datareader-dpl:$N_F_RAW | "
  has_detector HMP && WORKFLOW+="o2-hmpid-raw-to-digits-stream-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline HMP-RawStreamDecoder:$N_F_RAW | "
  has_detector CTP && WORKFLOW+="o2-ctp-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output --pipeline CTP-RawStreamDecoder:$N_CTPRAWDEC | "
  has_detector PHS && ! has_detector_flp_processing PHS && WORKFLOW+="o2-phos-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type raw --output-type cells --disable-root-input --disable-root-output --pipeline PHOSRawToCellConverterSpec:$N_F_REST $DISABLE_MC | "
  has_detector CPV && WORKFLOW+="o2-cpv-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type $CPV_INPUT --output-type clusters --disable-root-input --disable-root-output --pipeline CPVRawToDigitConverterSpec:$N_F_REST,CPVClusterizerSpec:$N_F_REST $DISABLE_MC | "
  has_detector EMC && ! has_detector_flp_processing EMC && WORKFLOW+="o2-emcal-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type raw --output-type cells --disable-root-output $DISABLE_MC --pipeline EMCALRawToCellConverterSpec:$N_EMC | "
fi

# ---------------------------------------------------------------------------------------------------------------------
# Common reconstruction workflows
(has_detector_reco TPC || has_detector_ctf TPC) && WORKFLOW+="o2-gpu-reco-workflow ${ARGS_ALL//-severity $SEVERITY/-severity $SEVERITY_TPC} --input-type=$GPU_INPUT $DISABLE_MC --output-type $GPU_OUTPUT --pipeline gpu-reconstruction:$N_TPCTRK $GPU_CONFIG --configKeyValues \"$ARGS_ALL_CONFIG;GPU_global.deviceType=$GPUTYPE;GPU_proc.debugLevel=0;$GPU_CONFIG_KEY;$GPU_EXTRA_CONFIG\" | "
(has_detector_reco TOF || has_detector_ctf TOF) && WORKFLOW+="o2-tof-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type $TOF_INPUT --output-type $TOF_OUTPUT --disable-root-input --disable-root-output $DISABLE_MC --pipeline tof-compressed-decoder:$N_F_RAW,TOFClusterer:$N_F_REST | "
has_detector_reco ITS && WORKFLOW+="o2-its-reco-workflow $ARGS_ALL --trackerCA $DISABLE_MC --clusters-from-upstream --disable-root-output $ITS_CONFIG --configKeyValues \"$ARGS_ALL_CONFIG;$ITS_CONFIG_KEY;$ITSMFT_FILES\" --pipeline its-tracker:$N_ITSTRK | "
has_detectors_reco ITS TPC && has_detector_matching ITSTPC && WORKFLOW+="o2-tpcits-match-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG;$ITSMFT_FILES\" --disable-root-input --disable-root-output $DISABLE_MC --pipeline itstpc-track-matcher:$N_TPCITS | "
has_detector_reco FT0 && WORKFLOW+="o2-ft0-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --pipeline ft0-reconstructor:$N_F_REST | "
has_detector_reco TRD && WORKFLOW+="o2-trd-tracklet-transformer $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC $TRD_TRANSFORMER_CONFIG --pipeline TRDTRACKLETTRANSFORMER:$N_TRDTRK | "
has_detectors_reco TRD TPC ITS && [ ! -z "$TRD_SOURCES" ] && WORKFLOW+="o2-trd-global-tracking $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG;$TRD_CONFIG_KEY\" --disable-root-input --disable-root-output $DISABLE_MC $TRD_CONFIG --track-sources $TRD_SOURCES | "
has_detectors_reco TOF TRD TPC ITS && [ ! -z "$TOF_SOURCES" ] && WORKFLOW+="o2-tof-matcher-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --track-sources $TOF_SOURCES --pipeline tof-matcher:$N_TOFMATCH | "

# ---------------------------------------------------------------------------------------------------------------------
# Reconstruction workflows in async mode
if [ $SYNCMODE == 0 ]; then
  has_detector_reco MID && WORKFLOW+="o2-mid-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output $DISABLE_MC --pipeline MIDClusterizer:$N_F_REST,MIDTracker:$N_F_REST | "
  has_detector_reco MCH && WORKFLOW+="o2-mch-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --pipeline mch-track-finder:$N_MCHTRK,mch-cluster-finder:$N_F_REST,mch-cluster-transformer:$N_F_REST | "
  has_detector_reco MFT && WORKFLOW+="o2-mft-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG;$ITSMFT_FILES\" --clusters-from-upstream $DISABLE_MC --disable-root-output --pipeline mft-tracker:$N_MFTTRK | "
  has_detector_reco FDD && WORKFLOW+="o2-fdd-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC | "
  has_detector_reco FV0 && WORKFLOW+="o2-fv0-reco-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC | "
  has_detector_reco ZDC && WORKFLOW+="o2-zdc-digits-reco $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC | "
fi

has_detectors_reco MFT MCH && has_detector_matching MFTMCH && WORKFLOW+="o2-globalfwd-matcher-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $DISABLE_MC --pipeline globalfwd-track-matcher:$N_F_REST | "
has_detectors_reco ITS && has_detector_matching PRIMVTX && WORKFLOW+="o2-primary-vertexing-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" $DISABLE_MC --disable-root-input --disable-root-output $PVERTEX_CONFIG --pipeline primary-vertexing:$N_F_REST | "
has_detectors_reco ITS && has_detector_matching SECVTX && WORKFLOW+="o2-secondary-vertexing-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-input --disable-root-output $SECVERTEX_CONFIG --pipeline secondary-vertexing:$N_F_REST | "

# ---------------------------------------------------------------------------------------------------------------------
# Entropy encoding / ctf creation workflows - disabled in async mode
if [ $CTFINPUT == 0 ] && [ ! -z "$WORKFLOW_DETECTORS_CTF" ]; then
  # Entropy encoder workflows
  has_detector_ctf MFT && WORKFLOW+="o2-itsmft-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --runmft true --pipeline mft-entropy-encoder:$N_F_CTF | "
  has_detector_ctf FT0 && WORKFLOW+="o2-ft0-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline ft0-entropy-encoder:$N_F_CTF | "
  has_detector_ctf FV0 && WORKFLOW+="o2-fv0-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline fv0-entropy-encoder:$N_F_CTF | "
  has_detector_ctf MID && WORKFLOW+="o2-mid-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline mid-entropy-encoder:$N_F_CTF | "
  has_detector_ctf MCH && WORKFLOW+="o2-mch-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline mch-entropy-encoder:$N_F_CTF | "
  has_detector_ctf PHS && WORKFLOW+="o2-phos-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline phos-entropy-encoder:$N_F_CTF | "
  has_detector_ctf CPV && WORKFLOW+="o2-cpv-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline cpv-entropy-encoder:$N_F_CTF | "
  has_detector_ctf EMC && WORKFLOW+="o2-emcal-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline emcal-entropy-encoder:$N_F_CTF | "
  has_detector_ctf ZDC && WORKFLOW+="o2-zdc-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline zdc-entropy-encoder:$N_F_CTF | "
  has_detector_ctf FDD && WORKFLOW+="o2-fdd-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline fdd-entropy-encoder:$N_F_CTF | "
  has_detector_ctf HMP && WORKFLOW+="o2-hmpid-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline hmpid-entropy-encoder:$N_F_CTF | "
  has_detector_ctf TOF && WORKFLOW+="o2-tof-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline tof-entropy-encoder:$N_F_CTF | "
  has_detector_ctf ITS && WORKFLOW+="o2-itsmft-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline its-entropy-encoder:$N_F_CTF | "
  has_detector_ctf TRD && WORKFLOW+="o2-trd-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline trd-entropy-encoder:$N_TRDENT | "
  has_detector_ctf TPC && WORKFLOW+="o2-tpc-reco-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --input-type compressed-clusters-flat --output-type encoded-clusters,disable-writer --pipeline tpc-entropy-encoder:$N_TPCENT | "
  has_detector_ctf CTP && WORKFLOW+="o2-ctp-entropy-encoder-workflow $ARGS_ALL --ctf-dict \"${CTF_DICT}\" --configKeyValues \"$ARGS_ALL_CONFIG\" --pipeline its-entropy-encoder:$N_F_CTF| "

  # CTF / dictionary writer workflow
  if [ $SAVECTF == 1 ]; then
    mkdir -p $CTF_DIR
  fi
  if [ $CREATECTFDICT == 1 ] ; then
    mkdir -p $CTF_DICT_DIR;
    rm -f $CTF_DICT
  fi
  CTF_OUTPUT_TYPE="none"
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="both"; fi
  if [ $CREATECTFDICT == 1 ] && [ $SAVECTF == 0 ]; then CTF_OUTPUT_TYPE="dict"; fi
  if [ $CREATECTFDICT == 0 ] && [ $SAVECTF == 1 ]; then CTF_OUTPUT_TYPE="ctf"; fi
  CMD_CTF="o2-ctf-writer-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --output-dir \"$CTF_DIR\" --ctf-dict-dir \"$CTF_DICT_DIR\" --output-type $CTF_OUTPUT_TYPE --min-file-size ${CTF_MINSIZE} --onlyDet $WORKFLOW_DETECTORS"
  if [ $CREATECTFDICT == 1 ] && [ $EXTINPUT == 1 ]; then CMD_CTF+=" --save-dict-after $NTIMEFRAMES"; fi
  WORKFLOW+="$CMD_CTF | "
fi

# ---------------------------------------------------------------------------------------------------------------------
# Calibration workflows
workflow_has_parameter CALIB && has_detector_calib TPC && has_detectors TPC ITS TRD TOF && WORKFLOW+="o2-tpc-scdcalib-interpolation-workflow $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG\" --disable-root-output --disable-root-input --pipeline tpc-track-interpolation:$N_F_REST | "

# ---------------------------------------------------------------------------------------------------------------------
# Event display
workflow_has_parameter EVENT_DISPLAY && [ $NUMAID == 0 ] && WORKFLOW+="o2-eve-display $ARGS_ALL --configKeyValues \"$ARGS_ALL_CONFIG;$ITSMFT_FILES\" --display-tracks TPC --display-clusters TPC $EVE_CONFIG $DISABLE_MC | "

# ---------------------------------------------------------------------------------------------------------------------
# Quality Control
workflow_has_parameter QC && source $MYDIR/qc-workflow.sh

# ---------------------------------------------------------------------------------------------------------------------
# DPL run binary
WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT"

# ---------------------------------------------------------------------------------------------------------------------
# Run / create / print workflow
if [ $WORKFLOWMODE == "print" ]; then
  echo "#Workflow command:"
  echo $WORKFLOW | sed "s/| */|\n/g"
else
  # Execute the command we have assembled
  WORKFLOW+=" --$WORKFLOWMODE"
  eval $WORKFLOW
fi

# ---------------------------------------------------------------------------------------------------------------------
