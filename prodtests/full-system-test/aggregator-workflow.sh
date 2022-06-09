#!/bin/bash

#SEVERITY="detail"
#ENABLE_METRICS=1
[ -d "$O2DPG_ROOT" ] || { echo "O2DPG_ROOT not set" 1>&2; exit 1; }

source $O2DPG_ROOT/DATA/common/setenv.sh
source $O2_ROOT/prodtests/full-system-test/workflow-setup.sh
source $O2DPG_ROOT/DATA/common/getCommonArgs.sh
source $O2DPG_ROOT/DATA/common/setenv_calib.sh

# check that WORKFLOW_DETECTORS is needed, otherwise the wrong calib wf will be built
if [[ -z $WORKFLOW_DETECTORS ]]; then echo "WORKFLOW_DETECTORS must be defined" 1>&2; exit 1; fi

# CCDB destination for uploads
[[ -z ${CCDB_POPULATOR_UPLOAD_PATH+x} ]] && CCDB_POPULATOR_UPLOAD_PATH="none"
echo "CCDB_POPULATOR_UPLOAD_PATH = $CCDB_POPULATOR_UPLOAD_PATH" 1>&2

# Adding calibrations
EXTRA_WORKFLOW_CALIB=

echo "CALIB_PRIMVTX_MEANVTX = $CALIB_PRIMVTX_MEANVTX" 1>&2
echo "CALIB_TOF_LHCPHASE = $CALIB_TOF_LHCPHASE" 1>&2
echo "CALIB_TOF_CHANNELOFFSETS = $CALIB_TOF_CHANNELOFFSETS" 1>&2
echo "CALIB_TOF_DIAGNOSTICS = $CALIB_TOF_DIAGNOSTICS" 1>&2
echo "CALIB_EMC_CHANNELCALIB = $CALIB_EMC_CHANNELCALIB" 1>&2
echo "CALIB_PHS_ENERGYCALIB = $CALIB_PHS_ENERGYCALIB" 1>&2
echo "CALIB_PHS_BADMAPCALIB = $CALIB_PHS_BADMAPCALIB" 1>&2
echo "CALIB_PHS_TURNONCALIB = $CALIB_PHS_TURNONCALIB" 1>&2
echo "CALIB_PHS_RUNBYRUNCALIB = $CALIB_PHS_RUNBYRUNCALIB" 1>&2
echo "CALIB_TRD_VDRIFTEXB = $CALIB_TRD_VDRIFTEXB" 1>&2
echo "CALIB_TPC_TIMEGAIN = $CALIB_TPC_TIMEGAIN" 1>&2
echo "CALIB_TPC_RESPADGAIN = $CALIB_TPC_RESPADGAIN" 1>&2

# beamtype dependent settings
LHCPHASE_TF_PER_SLOT=26400
TOF_CHANNELOFFSETS_UPDATE=300000
TOF_CHANNELOFFSETS_DELTA_UPDATE=50000

if [[ $BEAMTYPE == "PbPb" ]]; then
  LHCPHASE_TF_PER_SLOT=264
  TOF_CHANNELOFFSETS_UPDATE=3000
  TOF_CHANNELOFFSETS_DELTA_UPDATE=500
fi

# Calibration workflows
if ! workflow_has_parameter CALIB_LOCAL_INTEGRATED_AGGREGATOR; then WORKFLOW=; fi

# adding input proxies
if workflow_has_parameter CALIB_PROXIES; then
  if [[ ! -z $CALIBDATASPEC_BARREL ]]; then
    add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_BARREL\" $(get_proxy_connection barrel input)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_CALO ]]; then
    add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_CALO\" $(get_proxy_connection calo input)" "" 0
  fi
fi

# PrimVertex
if [[ $CALIB_PRIMVTX_MEANVTX == 1 ]]; then
  add_W o2-calibration-mean-vertex-calibration-workflow ""
fi

# TOF
if [[ $CALIB_TOF_LHCPHASE == 1 ]] || [[ $CALIB_TOF_CHANNELOFFSETS == 1 ]]; then
  if [[ $CALIB_TOF_LHCPHASE == 1 ]]; then
    add_W o2-calibration-tof-calib-workflow "--do-lhc-phase --tf-per-slot $LHCPHASE_TF_PER_SLOT --use-ccdb" "" 0
  fi
  if [[ $CALIB_TOF_CHANNELOFFSETS == 1 ]]; then
    add_W o2-calibration-tof-calib-workflow "--do-channel-offset --update-interval $TOF_CHANNELOFFSETS_UPDATE --delta-update-interval $TOF_CHANNELOFFSETS_DELTA_UPDATE --min-entries 100 --range 100000 --use-ccdb --follow-ccdb-updates" "" 0
  fi
fi
if [[ $CALIB_TOF_DIAGNOSTICS == 1 ]]; then
  add_W o2-calibration-tof-diagnostic-workflow "--tf-per-slot 26400 --max-delay 1" "" 0
fi

# TPC
if [[ $CALIB_TPC_TIMEGAIN == 1 ]]; then
  add_W o2-tpc-calibrator-dedx "--min-entries-sector 3000 --min-entries-1d 200 --min-entries-2d 10000"
fi
if [[ $CALIB_TPC_RESPADGAIN == 1 ]]; then
  add_W o2-tpc-calibrator-gainmap-tracks "--tf-per-slot 10000" "" 0
fi

# TRD
if [[ $CALIB_TRD_VDRIFTEXB == 1 ]]; then
  add_W o2-calibration-trd-vdrift-exb ""
fi

# Calo cal
# EMC
if [[ $CALIB_EMC_CHANNELCALIB == 1 ]]; then
  add_W o2-calibration-emcal-channel-calib-workflow "" "EMCALCalibParams.calibType=\"time\""
fi

# PHS
if [[ $CALIB_PHS_ENERGYCALIB == 1 ]]; then
  add_W o2-phos-calib-workflow "--energy"
fi
if [[ $CALIB_PHS_BADMAPCALIB == 1 ]]; then
  add_W o2-phos-calib-workflow "--badmap --mode 0"
fi
if [[ $CALIB_PHS_TURNONCALIB == 1 ]]; then
  add_W o2-phos-calib-workflow "--turnon"
fi
if [[ $CALIB_PHS_RUNBYRUNCALIB == 1 ]]; then
  add_W o2-phos-calib-workflow "--runbyrun"
fi

if [[ $CCDB_POPULATOR_UPLOAD_PATH != "none" ]]; then add_W o2-calibration-ccdb-populator-workflow "--ccdb-path $CCDB_POPULATOR_UPLOAD_PATH"; fi

if ! workflow_has_parameter CALIB_LOCAL_INTEGRATED_AGGREGATOR; then
  WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT"
  [[ $WORKFLOWMODE != "print" ]] && WORKFLOW+=" --${WORKFLOWMODE}"
  [[ $WORKFLOWMODE == "print" || "0$PRINT_WORKFLOW" == "01" ]] && echo "#Aggregator Workflow command:\n\n${WORKFLOW}\n" | sed -e "s/\\\\n/\n/g" -e"s/| */| \\\\\n/g" | eval cat $( [[ $WORKFLOWMODE == "dds" ]] && echo '1>&2')
  [[ $WORKFLOWMODE != "print" ]] && eval $WORKFLOW
fi
