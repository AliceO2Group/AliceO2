#!/bin/bash

source $O2DPG_ROOT/DATA/common/setenv.sh || { echo "setenv.sh failed" 1>&2 && exit 1; }
source $O2DPG_ROOT/DATA/common/setenv_calib.sh || { echo "setenv_calib.sh failed" 1>&2 && exit 1; }

if [[ -z "$WORKFLOW" ]] || [[ -z "$GEN_TOPO_MYDIR" ]]; then
  echo This script must be called from the dpl-workflow.sh and not standalone 1>&2
  exit 1
fi

# you cannot have a locally integrated aggregator with the proxies
if workflow_has_parameters CALIB_LOCAL_INTEGRATED_AGGREGATOR CALIB_PROXIES; then
  echo "you cannot have a locally integrated aggregator with the proxies" 1>&2
  exit 2
fi

if [[ -z ${TPC_IDCCALIB_NSLICESPERTF:-} ]]; then
  TPC_IDCCALIB_NSLICESPERTF=$(echo "scale=2;$NHBPERTF/128*11" | bc)
  TPC_IDCCALIB_NSLICESPERTF=$(echo $TPC_IDCCALIB_NSLICESPERTF | awk '{print int($1 + 0.5)}')
fi


if [[ "${CALIB_TPC_SCDCALIB_SENDTRKDATA:-}" == "1" ]]; then ENABLE_TRKDATA_OUTPUT="--send-track-data"; else ENABLE_TRKDATA_OUTPUT=""; fi

# specific calibration workflows
if [[ $CALIB_TPC_SCDCALIB == 1 ]]; then add_W o2-tpc-scdcalib-interpolation-workflow "--vtx-sources $VERTEX_TRACK_MATCHING_SOURCES ${CALIB_TPC_SCDCALIB_SLOTLENGTH:+"--sec-per-slot $CALIB_TPC_SCDCALIB_SLOTLENGTH"} $ENABLE_TRKDATA_OUTPUT $DISABLE_ROOT_OUTPUT --disable-root-input --pipeline $(get_N tpc-track-interpolation TPC REST)"; fi
if [[ $CALIB_TPC_TIMEGAIN == 1 ]]; then
  : ${SCALEEVENTS_TPC_TIMEGAIN:=40}
  : ${SCALETRACKS_TPC_TIMEGAIN:=1000}
  add_W o2-tpc-miptrack-filter "--processEveryNthTF $SCALEEVENTS_TPC_TIMEGAIN --maxTracksPerTF $SCALETRACKS_TPC_TIMEGAIN" "" 0
fi
if [[ $CALIB_TPC_RESPADGAIN == 1 ]]; then
  : ${SCALEEVENTS_TPC_RESPADGAIN:=40}
  add_W o2-tpc-calib-gainmap-tracks "--publish-after-tfs 30 --useEveryNthTF $SCALEEVENTS_TPC_RESPADGAIN  $TPC_CORR_SCALING"
fi
if [[ $CALIB_ZDC_TDC == 1 ]]; then add_W o2-zdc-tdccalib-epn-workflow "" "" 0; fi
if [[ $CALIB_FT0_TIMEOFFSET == 1 ]]; then add_W o2-calibration-ft0-time-spectra-processor; fi

# current integration
: ${INTEGRATED_CURR_OPTIONS:=""}
if [[ $SYNCMODE == 1 ]]; then INTEGRATED_CURR_OPTIONS="$DISABLE_ROOT_OUTPUT --nSlicesTF $TPC_IDCCALIB_NSLICESPERTF"; fi
if [[ $CALIB_FT0_INTEGRATEDCURR == 1 ]]; then add_W o2-ft0-integrate-cluster-workflow "$INTEGRATED_CURR_OPTIONS"; fi
if [[ $CALIB_FV0_INTEGRATEDCURR == 1 ]]; then add_W o2-fv0-integrate-cluster-workflow "$INTEGRATED_CURR_OPTIONS"; fi
if [[ $CALIB_FDD_INTEGRATEDCURR == 1 ]]; then add_W o2-fdd-integrate-cluster-workflow "$INTEGRATED_CURR_OPTIONS"; fi
if [[ $CALIB_TOF_INTEGRATEDCURR == 1 ]]; then add_W o2-tof-integrate-cluster-workflow "$INTEGRATED_CURR_OPTIONS"; fi

# for async calibrations
if [[ $CALIB_EMC_ASYNC_RECALIB == 1 ]]; then add_W o2-emcal-emc-offline-calib-workflow "--input-subspec 1 --applyGainCalib"; fi
: ${ARGS_EXTRA_PROCESS_o2_tpc_integrate_cluster_workflow:=""}
if [[ $CALIB_ASYNC_EXTRACTTPCCURRENTS == 1 ]]; then
  add_comma_separated ADD_EXTRA_WORKFLOW "o2-tpc-integrate-cluster-workflow"
  if [[ $CALIB_ASYNC_DISABLE3DCURRENTS != 1 ]]; then
    export ARGS_EXTRA_PROCESS_o2_tpc_integrate_cluster_workflow="$ARGS_EXTRA_PROCESS_o2_tpc_integrate_cluster_workflow--process-3D-currents --nSlicesTF 1"
  fi
fi
if [[ $CALIB_ASYNC_EXTRACTTIMESERIES == 1 ]] ; then
  : ${ARGS_EXTRA_PROCESS_o2_tpc_time_series_workflow:=""}
  : ${CALIB_ASYNC_SAMPLINGFACTORTIMESERIES:=0.001}
  add_comma_separated ADD_EXTRA_WORKFLOW "o2-tpc-time-series-workflow"
  if [[ ! -z "$CALIB_ASYNC_ENABLEUNBINNEDTIMESERIES" ]]; then
    export ARGS_EXTRA_PROCESS_o2_tpc_time_series_workflow="$ARGS_EXTRA_PROCESS_o2_tpc_time_series_workflow --enable-unbinned-root-output --sample-unbinned-tsallis --threads 1"
  fi
  if [[ $ON_SKIMMED_DATA == 1 ]] || [[ ! -z "$CALIB_ASYNC_SAMPLINGFACTORTIMESERIES" ]]; then
    if [[ $ON_SKIMMED_DATA == 1 ]]; then
      SAMPLINGFACTORTIMESERIES=0.1
    fi
    if [[ ! -z "$CALIB_ASYNC_SAMPLINGFACTORTIMESERIES" ]]; then # this takes priority, even if we were on skimmed data
      SAMPLINGFACTORTIMESERIES=${CALIB_ASYNC_SAMPLINGFACTORTIMESERIES}
    fi
    export ARGS_EXTRA_PROCESS_o2_tpc_time_series_workflow="$ARGS_EXTRA_PROCESS_o2_tpc_time_series_workflow --sampling-factor ${SAMPLINGFACTORTIMESERIES}"
  fi
fi

# output-proxy for aggregator
if workflow_has_parameter CALIB_PROXIES; then
  if [[ ! -z ${CALIBDATASPEC_BARREL_TF:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_BARREL_TF\" $(get_proxy_connection barrel_tf output timeframe)" "" 0
  fi
  if [[ ! -z ${CALIBDATASPEC_BARREL_SPORADIC:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_BARREL_SPORADIC\" $(get_proxy_connection barrel_sp output sporadic)" "" 0
  fi
  if [[ ! -z ${CALIBDATASPEC_CALO_TF:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_CALO_TF\" $(get_proxy_connection calo_tf output timeframe)" "" 0
  fi
  if [[ ! -z ${CALIBDATASPEC_CALO_SPORADIC:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_CALO_SPORADIC\" $(get_proxy_connection calo_sp output sporadic)" "" 0
  fi
  if [[ ! -z ${CALIBDATASPEC_MUON_TF:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_MUON_TF\" $(get_proxy_connection muon_tf output timeframe)" "" 0
  fi
  if [[ ! -z ${CALIBDATASPEC_MUON_SPORADIC:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_MUON_SPORADIC\" $(get_proxy_connection muon_sp output sporadic)" "" 0
  fi
  if [[ ! -z ${CALIBDATASPEC_FORWARD_TF:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_FORWARD_TF\" $(get_proxy_connection fwd_tf output timeframe)" "" 0
  fi
  if [[ ! -z ${CALIBDATASPEC_FORWARD_SPORADIC:-} ]]; then
    add_W o2-dpl-output-proxy "--dataspec \"$CALIBDATASPEC_FORWARD_SPORADIC\" $(get_proxy_connection fwd_sp output sporadic)" "" 0
  fi
fi

true # everything OK up to this point, so the script should return 0 (it is !=0 already if a has_detector check fails)
