#!/bin/bash

source $O2DPG_ROOT/DATA/common/setenv.sh
source $O2DPG_ROOT/DATA/common/setenv_calib.sh

if [[ -z "$WORKFLOW" ]] || [[ -z "$MYDIR" ]]; then
  echo This script must be called from the dpl-workflow.sh and not standalone 1>&2
  exit 1
fi

# you cannot have a locally integrated aggregator with the proxies
if workflow_has_parameters CALIB_LOCAL_INTEGRATED_AGGREGATOR CALIB_PROXIES; then
  echo "you cannot have a locally integrated aggregator with the proxies" 1>&2
  exit 2
fi

if [[ "0$CALIB_TPC_SCDCALIB_SENDTRKDATA" == "01" ]]; then ENABLE_TRKDATA_OUTPUT="--send-track-data"; fi

# specific calibration workflows
if [[ $CALIB_TPC_SCDCALIB == 1 ]]; then add_W o2-tpc-scdcalib-interpolation-workflow "$ENABLE_TRKDATA_OUTPUT $DISABLE_ROOT_OUTPUT --disable-root-input --pipeline $(get_N tpc-track-interpolation TPC REST)" "$ITSMFT_FILES"; fi
if [[ $CALIB_TPC_TIMEGAIN == 1 ]]; then
  if [[ -z $SCALEEVENTS_TPC_TIMEGAIN ]]; then SCALEEVENTS_TPC_TIMEGAIN=10; fi
  if [[ -z $SCALETRACKS_TPC_TIMEGAIN ]]; then SCALETRACKS_TPC_TIMEGAIN=1000; fi
  add_W o2-tpc-miptrack-filter "--processEveryNthTF $SCALEEVENTS_TPC_TIMEGAIN --maxTracksPerTF $SCALETRACKS_TPC_TIMEGAIN" "" 0
fi
if [[ $CALIB_TPC_RESPADGAIN == 1 ]]; then add_W o2-tpc-calib-gainmap-tracks "--publish-after-tfs 30"; fi
if [[ $CALIB_ZDC_TDC == 1 ]]; then add_W o2-zdc-tdccalib-epn-workflow "" "" 0; fi
if [[ $CALIB_FT0_TIMEOFFSET == 1 ]]; then add_W o2-calibration-ft0-time-spectra-processor; fi
# for async calibrations
if [[ $CALIB_EMC_ASYNC_RECALIB == 1 ]]; then add_W o2-emcal-emc-offline-calib-workflow; fi

# output-proxy for aggregator
if workflow_has_parameter CALIB_PROXIES; then
  if [[ ! -z $CALIBDATASPEC_BARREL_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_BARREL_TF\" $(get_proxy_connection barrel_tf output timeframe)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_BARREL_SPORADIC ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_BARREL_SPORADIC\" $(get_proxy_connection barrel_sp output sporadic)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_CALO_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_CALO_TF\" $(get_proxy_connection calo_tf output timeframe)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_CALO_SPORADIC ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_CALO_SPORADIC\" $(get_proxy_connection calo_sp output sporadic)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_MUON_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_MUON_TF\" $(get_proxy_connection muon_tf output timeframe)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_MUON_SPORADIC ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_MUON_SPORADIC\" $(get_proxy_connection muon_sp output sporadic)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_FORWARD_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_FORWARD_TF\" $(get_proxy_connection fwd_tf output timeframe)" "" 0
  fi

fi

true # everything OK up to this point, so the script should return 0 (it is !=0 already if a has_detector check fails)
