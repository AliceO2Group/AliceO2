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

# specific calibration workflows
if [[ $CALIB_TPC_SCDCALIB == 1 ]]; then add_W o2-tpc-scdcalib-interpolation-workflow "$DISABLE_ROOT_OUTPUT --disable-root-input --pipeline $(get_N tpc-track-interpolation TPC REST)" "$ITSMFT_FILES"; fi
if [[ $CALIB_TPC_TIMEGAIN == 1 ]]; then add_W o2-tpc-miptrack-filter "" "" 0; fi
if [[ $CALIB_TPC_RESPADGAIN == 1 ]]; then add_W o2-tpc-calib-gainmap-tracks "--publish-after-tfs 10000"; fi

# output-proxy for aggregator
if workflow_has_parameter CALIB_PROXIES; then
  if [[ ! -z $CALIBDATASPEC_BARREL ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_BARREL\" $(get_proxy_connection barrel output)" "" 0
  fi
  if [[ ! -z $CALIBDATASPEC_CALO ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_CALO\" $(get_proxy_connection calo output)" "" 0
  fi
fi

true # everything OK up to this point, so the script should return 0 (it is !=0 already if a has_detector check fails)
