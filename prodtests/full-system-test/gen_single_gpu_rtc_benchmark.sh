#!/usr/bin/env bash
# Generate or run a single-GPU reconstruction benchmark workflow using dpl-workflow.sh.
#
# Main benchmark mode:
#   RUN_BENCHMARK=1 GPUTYPE=HIP FILEWORKDIR=/path/to/raw_tf_dir ./gen_single_gpu_rtc_benchmark.sh

set -euo pipefail

rm -rf /dev/shm/fmq*

: "${O2_DPL_WORKFLOW:=$O2_ROOT/prodtests/full-system-test/dpl-workflow.sh}"

if [[ ! -f "$O2_DPL_WORKFLOW" ]]; then
  echo "FATAL: dpl workflow script does not exist: $O2_DPL_WORKFLOW" >&2
  echo "Set O2_DPL_WORKFLOW=/path/to/dpl-workflow.sh" >&2
  exit 1
fi

case "${GPUTYPE:-}" in
  CUDA|HIP|CPU|OPENCL)
    export GPUTYPE
    ;;
  "")
    echo "ERROR: GPUTYPE must be set to one of: CUDA, HIP, CPU, OPENCL" >&2
    exit 1
    ;;
  *)
    echo "ERROR: Invalid GPUTYPE='$GPUTYPE'. Must be one of: CUDA, HIP, CPU, OPENCL" >&2
    exit 1
    ;;
esac

if [[ -z "${FILEWORKDIR:-}" && -z "${INPUT_FILE_LIST:-}" ]]; then
  echo "ERROR: either FILEWORKDIR or INPUT_FILE_LIST must be set" >&2
  exit 1
fi

if [[ -n "${FILEWORKDIR:-}" && "$FILEWORKDIR" != /* ]]; then
  echo "ERROR: FILEWORKDIR must be an absolute path: $FILEWORKDIR" >&2
  exit 1
fi

if [[ -n "${INPUT_FILE_LIST:-}" && "$INPUT_FILE_LIST" != /* ]]; then
  echo "ERROR: INPUT_FILE_LIST must be an absolute path: $INPUT_FILE_LIST" >&2
  exit 1
fi

# ----------------------------------------------------------------------------------------------------------------------
# Benchmark defaults. All can be overridden by exporting variables before calling this script.

export DPL_REPORT_PROCESSING="${DPL_REPORT_PROCESSING:-1}"
export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS:-GPU,CTF}"
export NGPUS="${NGPUS:-1}"
export O2_GPU_DOUBLE_PIPELINE="${O2_GPU_DOUBLE_PIPELINE:-1}"
export O2_GPU_RTC="${O2_GPU_RTC:-1}"

# Reuse GPU RTC compilation cache by default.
export ENABLE_RTCCACHE_DIR="${ENABLE_RTCCACHE_DIR:-/tmp/rtc_cache}"
if [[ -n "${ENABLE_RTCCACHE_DIR:-}" ]]; then
  mkdir -p "$ENABLE_RTCCACHE_DIR"
  export CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow="${CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow:-}"
  CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow+="GPU_proc_rtc.cacheOutput=1;GPU_proc_rtctech.cacheFolder=${ENABLE_RTCCACHE_DIR};"
  export CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow
fi

export MULTITHREADING_CPU_PROCESSES="${MULTITHREADING_CPU_PROCESSES:-1}"
export MULTIPLICITY_PROCESS_its_tracker="${MULTIPLICITY_PROCESS_its_tracker:-$MULTITHREADING_CPU_PROCESSES}"
export MULTIPLICITY_PROCESS_itstpc_track_matcher="${MULTIPLICITY_PROCESS_itstpc_track_matcher:-$MULTITHREADING_CPU_PROCESSES}"
export ITSTRK_THREADS="${ITSTRK_THREADS:-$MULTITHREADING_CPU_PROCESSES}"
export ITSTPC_THREADS="${ITSTPC_THREADS:-$MULTITHREADING_CPU_PROCESSES}"

# Double pipeline requires zsraw input. Therefore default to raw TF input, not CTF.
export RAWTFINPUT="${RAWTFINPUT:-1}"
export SYNCMODE="${SYNCMODE:-1}"
export NTIMEFRAMES="${NTIMEFRAMES:--1}"
export TFLOOP="${TFLOOP:-100}"
export TFDELAY="${TFDELAY:-0.1}"
export TIMEFRAME_RATE_LIMIT="${TIMEFRAME_RATE_LIMIT:-10}"
export ARGS_EXTRA_PROCESS_o2_gpu_reco_workflow="${ARGS_EXTRA_PROCESS_o2_gpu_reco_workflow:+$ARGS_EXTRA_PROCESS_o2_gpu_reco_workflow }--log-timestamp-us"

export RUN_BENCHMARK="${RUN_BENCHMARK:-0}"

# ----------------------------------------------------------------------------------------------------------------------
# Benchmark naming / output directory.

: "${BENCH_TAG:=${BENCH_TAG:-$(hostname -s)}}"
BENCH_STAMP="$(date +%Y%m%d_%H%M%S)"
: "${OUTDIR:=${BENCHMARK_OUTDIR:-$PWD/single_gpu_rtc_bench_${BENCH_TAG}_${BENCH_STAMP}}}"

mkdir -p "$OUTDIR"
RUNDIR="$OUTDIR/run"
mkdir -p "$RUNDIR"

cleanup_rundir() {
  if [[ -n "${RUNDIR:-}" && -d "$RUNDIR" ]]; then
    echo "# Cleaning run dir: $RUNDIR"
    rm -rf -- "$RUNDIR"
  fi
  rm -rf /dev/shm/fmq*
}

trap cleanup_rundir EXIT

# Avoid copying input files unless the caller explicitly requests a copy command.
if [[ "${BENCH_DISABLE_INPUT_COPY:-1}" == "1" ]]; then
  unset INPUT_FILE_COPY_CMD || true
fi

# A single-GPU benchmark must not enter EPN sync mode, because the workflow intentionally sets NGPUS=4 there.
if [[ "${EPNSYNCMODE:-0}" == "1" ]]; then
  echo "FATAL: EPNSYNCMODE=1 is incompatible with the single-GPU RTC benchmark." >&2
  echo "EPNSYNCMODE=1 makes dpl-workflow.sh set GPUTYPE=HIP and NGPUS=4 by design." >&2
  echo "Use EPNSYNCMODE=0 for this benchmark." >&2
  exit 1
fi

# ----------------------------------------------------------------------------------------------------------------------
# Print configuration.

echo "# single-GPU RTC benchmark"
echo "# source script: $O2_DPL_WORKFLOW"
echo "# output dir:    $OUTDIR"
echo "# run dir:       $RUNDIR"
echo "# run benchmark: $RUN_BENCHMARK (0: prints workflow, 1: runs workflow)"
echo "# NGPUS=$NGPUS GPUTYPE=$GPUTYPE"
echo "# O2_GPU_DOUBLE_PIPELINE=$O2_GPU_DOUBLE_PIPELINE O2_GPU_RTC=$O2_GPU_RTC"
echo "# NTIMEFRAMES=$NTIMEFRAMES TFLOOP=$TFLOOP"
echo "# FILEWORKDIR=${FILEWORKDIR:-}"
echo "# RTC cache dir: ${ENABLE_RTCCACHE_DIR:-disabled}"
echo

# ----------------------------------------------------------------------------------------------------------------------
# Generate workflow with the caller-provided environment.

export WORKFLOWMODE="print"
cmdfile="$OUTDIR/workflow_${BENCH_TAG}_${BENCH_STAMP}.sh"

echo "# Generating workflow only; command file: $cmdfile"

(
  cd "$RUNDIR"
  "$O2_DPL_WORKFLOW"
) > "$cmdfile"

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  export WORKFLOWMODE="run"

  log="$OUTDIR/reco_${BENCH_TAG}_${BENCH_STAMP}.log"
  env | sort > "$OUTDIR/env_${BENCH_TAG}_${BENCH_STAMP}.txt"

  echo "# Running benchmark; log: $log"

  set +e
  (
    cd "$RUNDIR"
    chmod +x "$cmdfile"
    /usr/bin/time -v "$cmdfile"
  ) > "$log" 2>&1
  status=$?
  set -e

  echo "# Full log: $log"

  # --------------------------------------------------------------------------------------------------------------------
  # Analyze gpu-reconstruction processing timeslice timing and write PDF next to the log.

  : "${GPU_RECO_ANALYZER:=$O2_ROOT/prodtests/full-system-test/analyze_gpu_benchmarks.py}"

  if [[ -f "$GPU_RECO_ANALYZER" ]]; then
    analysis_pdf="${log%.log}_gpu_reconstruction_times.pdf"
    summary_txt="${log%.log}_gpu_reconstruction_summary.txt"

    echo "# Analyzing gpu-reconstruction timeslices"
    echo "# analyzer: $GPU_RECO_ANALYZER"
    echo "# plot:     $analysis_pdf"

    python3 "$GPU_RECO_ANALYZER" --logfile "$log" --output "$analysis_pdf" --summary-output "$summary_txt" || {
      echo "WARNING: gpu-reconstruction timing analysis failed" >&2
    }
  else
    echo "WARNING: gpu-reconstruction analyzer not found: $GPU_RECO_ANALYZER" >&2
  fi

  if [[ "$status" -ne 0 ]]; then
    echo -e "\033[31m------\nWARNING: reconstruction workflow exited with code $status\n-----\033[0m" >&2
  fi

  exit "$status"
fi