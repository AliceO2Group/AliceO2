#!/bin/bash

SEVERITY="detail"
ENABLE_METRICS=1
source $O2DPG_ROOT/DATA/common/setenv.sh
source $O2_ROOT/prodtests/full-system-test/workflow-setup.sh

# ---------------------------------------------------------------------------------------------------------------------
# Set general arguments
source $O2DPG_ROOT/DATA/common/getCommonArgs.sh

# Set up calibrations
source $O2DPG_ROOT/DATA/common/setenv_calib.sh

# Adding calibrations
EXTRA_WORKFLOW=

echo "CALIB_PRIMVTX_MEANVTX = $CALIB_PRIMVTX_MEANVTX" 1>&2
echo "CALIB_TOF_LHCPHASE = $CALIB_TOF_LHCPHASE" 1>&2
echo "CALIB_TOF_CHANNELOFFSETS = $CALIB_TOF_CHANNELOFFSETS" 1>&2
echo "CALIB_TOF_DIAGNOSTICS = $CALIB_TOF_DIAGNOSTICS" 1>&2

# PrimVertex
if [[ $CALIB_PRIMVTX_MEANVTX == 1 ]]; then
    EXTRA_WORKFLOW+="o2-calibration-mean-vertex-calibration-workflow $ARGS_ALL | "
fi

# TOF
if [[ $CALIB_TOF_LHCPHASE == 1 ]] || [[ $CALIB_TOF_CHANNELOFFSETS == 1 ]]; then
    if [[ $CALIB_TOF_LHCPHASE == 1 ]]; then
  EXTRA_WORKFLOW+="o2-calibration-tof-calib-workflow $ARGS_ALL --do-lhc-phase --tf-per-slot 10 | "
    fi
    if [[ $CALIB_TOF_CHANNELOFFSETS == 1 ]]; then
  EXTRA_WORKFLOW+="o2-calibration-tof-calib-workflow $ARGS_ALL --do-channel-offset --update-at-end-of-run-only --min-entries 8 --range 100000 |"
    fi
fi
if [[ $CALIB_TOF_DIAGNOSTICS == 1 ]]; then
    EXTRA_WORKFLOW+="o2-calibration-tof-diagnostic-workflow $ARGS_ALL --tf-per-slot 26400 | "
fi

WORKFLOW="o2-dpl-raw-proxy $ARGS_ALL --proxy-name aggregator-proxy --dataspec \"$CALIBDATASPEC\" --channel-config \"name=aggregator-proxy,method=bind,type=pull,rateLogging=1,transport=zeromq,address=tcp://localhost:30453\" |"
WORKFLOW+=$EXTRA_WORKFLOW
WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT"

if [ $WORKFLOWMODE == "print" ]; then
  echo Workflow command:
  echo $WORKFLOW | sed "s/| */|\n/g"
else
  # Execute the command we have assembled
  WORKFLOW+=" --$WORKFLOWMODE"
  eval $WORKFLOW
fi
