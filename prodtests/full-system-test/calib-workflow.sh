#!/bin/bash

# ---------------------------------------------------------------------------------------------------------------------
# Get this script's directory and load common settings (use zsh first (e.g. on Mac) and fallback on `readlink -f` if zsh is not there)
MYDIR="$(dirname $(realpath $0))"
source $O2DPG_ROOT/DATA/common/setenv.sh
source $O2DPG_ROOT/DATA/common/setenv_calib.sh

if [[ -z "$WORKFLOW" ]] || [[ -z "$MYDIR" ]]; then
  echo This script must be called from the dpl-workflow.sh and not standalone 1>&2
  exit 1
fi

# specific calibration workflows
if [[ $CALIB_TPC_SCDCALIB == 1 ]]; then add_W o2-tpc-scdcalib-interpolation-workflow "$DISABLE_ROOT_OUTPUT --disable-root-input --pipeline $(get_N tpc-track-interpolation TPC REST)" "$ITSMFT_FILES"; fi

# output-proxy for aggregator
if workflow_has_parameters CALIB_PROXIES; then
    if [[ ! -z $CALIBDATASPEC_BARREL ]]; then
  WORKFLOW+="o2-dpl-output-proxy ${ARGS_ALL} --dataspec \"$CALIBDATASPEC_BARREL\" $(get_proxy_connection barrel output) | "
    fi
    if [[ ! -z $CALIBDATASPEC_CALO ]]; then
  WORKFLOW+="o2-dpl-output-proxy ${ARGS_ALL} --dataspec \"$CALIBDATASPEC_CALO\" $(get_proxy_connection calo output) | "
    fi
fi

true # everything OK up to this point, so the script should return 0 (it is !=0 already if a has_detector check fails)
