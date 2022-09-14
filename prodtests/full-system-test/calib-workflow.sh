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
if [[ $CALIB_TPC_TIMEGAIN == 1 ]]; then add_W o2-tpc-miptrack-filter "" "" 0; fi
if [[ $CALIB_TPC_RESPADGAIN == 1 ]]; then add_W o2-tpc-calib-gainmap-tracks "--publish-after-tfs 10000"; fi
if [[ $CALIB_ZDC_TDC == 1 ]]; then add_W o2-zdc-tdccalib-epn-workflow "" "" 0; fi

# output-proxy for aggregator
if workflow_has_parameter CALIB_PROXIES; then
  if [[ ! -z $CALIBDATASPEC_BARREL_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_BARREL_TF\" $(get_proxy_connection barrel_tf output)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_BARREL_SPORADIC ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_BARREL_SPORADIC\" $(get_proxy_connection barrel_sp output)" "" 0
  fi
  if [[ ! -z $CALIB_TPC_IDC_BOTH ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_TPCIDC_A;$CALIBDATASPEC_TPCIDC_C\" $(get_proxy_connection tpcidc_both output)" "" 0
  else
    if [[ ! -z $CALIBDATASPEC_TPCIDC_A ]]; then
      add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_TPCIDC_A\" $(get_proxy_connection tpcidc_A output)" "" 0
    fi
    if [[ ! -z $CALIBDATASPEC_TPCIDC_C ]]; then
      add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_TPCIDC_C\" $(get_proxy_connection tpcidc_C output)" "" 0
    fi
  fi
  if [[ ! -z $CALIBDATASPEC_CALO_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_CALO_TF\" $(get_proxy_connection calo_tf output)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_CALO_SPORADIC ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_CALO_SPORADIC\" $(get_proxy_connection calo_sp output)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_MUON_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_MUON_TF\" $(get_proxy_connection muon_tf output)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_MUON_SPORADIC ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_MUON_SPORADIC\" $(get_proxy_connection muon_sp output)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_ZDC_TF ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_ZDC_TF\" $(get_proxy_connection zdc_tf output)" "" 0
  fi

fi

true # everything OK up to this point, so the script should return 0 (it is !=0 already if a has_detector check fails)
