#!/bin/bash

if [[ -z "$WORKFLOW" ]] || [[ -z "$MYDIR" ]]; then
  echo This script must be called from the dpl-workflow.sh and not standalone 1>&2
  exit 1
fi

has_detector_calib TPC && has_detectors TPC ITS TRD TOF && add_W o2-tpc-scdcalib-interpolation-workflow "$DISABLE_ROOT_OUTPUT --disable-root-input --pipeline $(get_N tpc-track-interpolation TPC REST)" "$ITSMFT_FILES"

true # everything OK up to this point, so the script should return 0 (it is !=0 already if a has_detector check fails)
