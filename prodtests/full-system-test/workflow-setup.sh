#!/bin/bash

# used to avoid sourcing this file 2x
if [[ -z $SOURCE_GUARD_WORKFLOW_SETUP ]]; then
SOURCE_GUARD_WORKFLOW_SETUP=1

# ---------------------------------------------------------------------------------------------------------------------
#Some additional settings used in this workflow

if [[ $SYNCMODE == 1 ]]; then
  if [[ -z "${WORKFLOW_DETECTORS_MATCHING+x}" ]]; then export WORKFLOW_DETECTORS_MATCHING="ITSTPC,ITSTPCTRD,ITSTPCTOF,ITSTPCTRDTOF,PRIMVTX"; fi # Select matchings that are enabled in sync mode
else
  if [[ -z "${WORKFLOW_DETECTORS_MATCHING+x}" ]]; then export WORKFLOW_DETECTORS_MATCHING="ALL"; fi # All matching / vertexing enabled in async mode
fi

MID_FEEID_MAP="$FILEWORKDIR/mid-feeId_mapper.txt"

ITSMFT_STROBES=""
[[ ! -z $ITS_STROBE ]] && ITSMFT_STROBES+="ITSAlpideParam.roFrameLengthInBC=$ITS_STROBE;"
[[ ! -z $MFT_STROBE ]] && ITSMFT_STROBES+="MFTAlpideParam.roFrameLengthInBC=$MFT_STROBE;"

LIST_OF_ASYNC_RECO_STEPS="MID MCH MFT FDD FV0 ZDC HMP"

DISABLE_DIGIT_ROOT_INPUT="--disable-root-input"
DISABLE_DIGIT_CLUSTER_INPUT="--clusters-from-upstream"

# ---------------------------------------------------------------------------------------------------------------------
# Set active reconstruction steps (defaults added according to SYNCMODE)

has_processing_step()
{
  [[ $WORKFLOW_EXTRA_PROCESSING_STEPS =~ (^|,)"$1"(,|$) ]]
}

for i in `echo $LIST_OF_GLORECO | sed "s/,/ /g"`; do
  has_processing_step MATCH_$i && add_comma_separated WORKFLOW_DETECTORS_MATCHING $i # Enable extra matchings requested via WORKFLOW_EXTRA_PROCESSING_STEPS
done
if [[ $SYNCMODE == 1 ]]; then # Add default steps for synchronous mode
  add_comma_separated WORKFLOW_EXTRA_PROCESSING_STEPS ENTROPY_ENCODER
else # Add default steps for async mode
  for i in $LIST_OF_ASYNC_RECO_STEPS; do
    has_detector_reco $i && add_comma_separated WORKFLOW_EXTRA_PROCESSING_STEPS ${i}_RECO
  done
  add_comma_separated WORKFLOW_EXTRA_PROCESSING_STEPS TPC_DEDX
fi

# ---------------------------------------------------------------------------------------------------------------------
# Assemble matching sources

TRD_SOURCES=
TOF_SOURCES=
TRACK_SOURCES=
has_detectors_reco ITS TPC && has_detector_matching ITSTPC && add_comma_separated TRACK_SOURCES "ITS-TPC"
has_detectors_reco TPC TRD && has_detector_matching TPCTRD && { add_comma_separated TRD_SOURCES TPC; add_comma_separated TRACK_SOURCES "TPC-TRD"; }
has_detectors_reco ITS TPC TRD && has_detector_matching ITSTPCTRD && { add_comma_separated TRD_SOURCES ITS-TPC; add_comma_separated TRACK_SOURCES "ITS-TPC-TRD"; }
has_detectors_reco TPC TOF && has_detector_matching TPCTOF && { add_comma_separated TOF_SOURCES TPC; add_comma_separated TRACK_SOURCES "TPC-TOF"; }
has_detectors_reco ITS TPC TOF && has_detector_matching ITSTPC && has_detector_matching ITSTPCTOF && { add_comma_separated TOF_SOURCES ITS-TPC; add_comma_separated TRACK_SOURCES "ITS-TPC-TOF"; }
has_detectors_reco TPC TRD TOF && has_detector_matching TPCTRD && has_detector_matching TPCTRDTOF && { add_comma_separated TOF_SOURCES TPC-TRD; add_comma_separated TRACK_SOURCES "TPC-TRD-TOF"; }
has_detectors_reco ITS TPC TRD TOF && has_detector_matching ITSTPCTRD && has_detector_matching ITSTPCTRDTOF && { add_comma_separated TOF_SOURCES ITS-TPC-TRD; add_comma_separated TRACK_SOURCES "ITS-TPC-TRD-TOF"; }
has_detectors_reco MFT MCH && has_detector_matching MFTMCH && add_comma_separated TRACK_SOURCES "MFT-MCH"
has_detectors_reco MCH MID && has_detector_matching MCHMID && add_comma_separated TRACK_SOURCES "MCH-MID"
for det in `echo $LIST_OF_DETECTORS | sed "s/,/ /g"`; do
  if [[ $LIST_OF_ASYNC_RECO_STEPS =~ (^| )${det}( |$) ]]; then
    has_detector ${det} && has_processing_step ${det}_RECO && add_comma_separated TRACK_SOURCES "$det"
  else
    has_detector_reco $det && add_comma_separated TRACK_SOURCES "$det"
  fi
done
[[ -z $VERTEXING_SOURCES ]] && VERTEXING_SOURCES="$TRACK_SOURCES"
PVERTEX_CONFIG="--vertexing-sources $VERTEXING_SOURCES --vertex-track-matching-sources $VERTEXING_SOURCES"

# this option requires well calibrated timing beween different detectors, at the moment suppress it
#has_detector_reco FT0 && PVERTEX_CONFIG+=" --validate-with-ft0"

# ---------------------------------------------------------------------------------------------------------------------
# Helper functions for multiplicities

get_N() # USAGE: get_N [processor-name] [DETECTOR_NAME] [RAW|CTF|REST] [threads, to be used for process scaling. 0 = do not scale this one process] [optional name [FOO] of variable "$N_[FOO]" with default, default = 1]
{
  local NAME_FACTOR="N_F_$3"
  local NAME_DET="MULTIPLICITY_FACTOR_DETECTOR_$2"
  local NAME_PROC="MULTIPLICITY_FACTOR_PROCESS_${1//-/_}"
  local NAME_DEFAULT="N_$5"
  local MULT=${!NAME_PROC:-$((${!NAME_FACTOR} * ${!NAME_DET:-1} * ${!NAME_DEFAULT:-1}))}
  if [[ "0$GEN_TOPO_AUTOSCALE_PROCESSES" == "01" && $WORKFLOWMODE != "print" && $4 != 0 ]]; then
    echo $1:\$\(\(\($MULT*\$AUTOSCALE_PROCESS_FACTOR/100\) \< 16 ? \($MULT*\$AUTOSCALE_PROCESS_FACTOR/100\) : 16\)\)
  else
    echo $1:$MULT
  fi
}

math_max()
{
  echo $(($1 > $2 ? $1 : $2))
}

# ---------------------------------------------------------------------------------------------------------------------
# Helper to add binaries to workflow adding automatic and custom arguments

add_W() # Add binarry to workflow command USAGE: add_W [BINARY] [COMMAND_LINE_OPTIONS] [CONFIG_KEY_VALUES] [Add ARGS_ALL_CONFIG, optional, default = 1]
{
  local NAME_PROC_ARGS="ARGS_EXTRA_PROCESS_${1//-/_}"
  local NAME_PROC_CONFIG="CONFIG_EXTRA_PROCESS_${1//-/_}"
  local KEY_VALUES=
  [[ "0$4" != "00" ]] && KEY_VALUES+="$ARGS_ALL_CONFIG;"
  [[ ! -z "$3" ]] && KEY_VALUES+="$3;"
  [[ ! -z ${!NAME_PROC_CONFIG} ]] && KEY_VALUES+="${!NAME_PROC_CONFIG};"
  [[ ! -z "$KEY_VALUES" ]] && KEY_VALUES="--configKeyValues \"$KEY_VALUES\""
  WORKFLOW+="$1 $ARGS_ALL $2 ${!NAME_PROC_ARGS} $KEY_VALUES | "
}

fi # workflow-setup.sh sourced
