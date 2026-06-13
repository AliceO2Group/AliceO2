#!/usr/bin/env bash
# Generate or run a single-GPU reconstruction benchmark workflow using dpl-workflow.sh.
#
# Main benchmark mode:
#   BENCHMARK_RUN=1 FILEWORKDIR=/path/to/raw_tf_dir ./gen_single_gpu_rtc_benchmark.sh

set -euo pipefail

rm -rf /dev/shm/fmq*

: "${O2_DPL_WORKFLOW:=$O2_ROOT/prodtests/full-system-test/dpl-workflow.sh}"

if [[ ! -f "$O2_DPL_WORKFLOW" ]]; then
  echo "FATAL: dpl workflow script does not exist: $O2_DPL_WORKFLOW" >&2
  echo "Set O2_DPL_WORKFLOW=/path/to/dpl-workflow.sh" >&2
  exit 1
fi

# ----------------------------------------------------------------------------------------------------------------------
# Benchmark defaults. All can be overridden by exporting variables before calling this script.

case "${GPUTYPE:-}" in
  CUDA|HIP)
    export GPUTYPE
    ;;
  "")
    echo "ERROR: GPUTYPE must be set to either CUDA or HIP" >&2
    exit 1
    ;;
  *)
    echo "ERROR: Invalid GPUTYPE='$GPUTYPE'. Must be either CUDA or HIP" >&2
    exit 1
    ;;
esac

export DPL_REPORT_PROCESSING="${DPL_REPORT_PROCESSING:-1}"
export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS:-GPU,CTF}"
export NGPUS=1
export O2_GPU_DOUBLE_PIPELINE="${O2_GPU_DOUBLE_PIPELINE:-1}"
export O2_GPU_RTC="${O2_GPU_RTC:-1}"
export SYNCMODE="${SYNCMODE:-1}"

# Double pipeline requires zsraw input. Therefore default to raw TF input, not CTF.
export RAWTFINPUT="${RAWTFINPUT:-1}"

export NTIMEFRAMES="${NTIMEFRAMES:--1}"
export TFLOOP="${TFLOOP:-100}"
export TFDELAY="${TFDELAY:-0}"
export TIMEFRAME_RATE_LIMIT="${TIMEFRAME_RATE_LIMIT:-5}"
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
}

trap cleanup_rundir EXIT

# Let O2/core dumps land in the benchmark run directory, not in the original working directory.
export CORE_DUMP_DIR="${CORE_DUMP_DIR:-$RUNDIR}"
export O2_CORE_DUMP_DIR="${O2_CORE_DUMP_DIR:-$RUNDIR}"

# Avoid copying input files unless the caller explicitly requests a copy command.
if [[ "${BENCH_DISABLE_INPUT_COPY:-1}" == "1" ]]; then
  unset INPUT_FILE_COPY_CMD || true
fi

# ----------------------------------------------------------------------------------------------------------------------
# Library path fixes for common EPN/dev-node issues.

: "${BENCH_USE_SYSTEM_FONT_LIBS:=1}"
: "${BENCH_AUTO_ROCM_LIBS:=0}"

prepend_ld_path() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$dir:"*) ;;
    *) export LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH:-}" ;;
  esac
}

if [[ "0$BENCH_USE_SYSTEM_FONT_LIBS" == "01" ]]; then
  prepend_ld_path /usr/lib64
  prepend_ld_path /lib64
fi

# ROCm library injection is only useful for HIP runs. Keep it off by default for CUDA/NVIDIA containers,
# because mixed AMD/NVIDIA hosts can otherwise leak ROCm libraries into LD_LIBRARY_PATH.
if [[ "${GPUTYPE:-}" == "HIP" && $BENCH_AUTO_ROCM_LIBS == 1 ]]; then
  if [[ -n "${ROCM_PATH:-}" ]]; then
    prepend_ld_path "$ROCM_PATH/lib64"
    prepend_ld_path "$ROCM_PATH/lib"
  fi
  for d in /opt/rocm/lib /opt/rocm/lib64 /usr/lib64/rocm /usr/lib/rocm/lib; do
    prepend_ld_path "$d"
  done
fi

if [[ -n "${BENCH_EXTRA_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$BENCH_EXTRA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
fi

# Check CUDA runtime/device visibility before starting the full workflow.
if [[ "$GPUTYPE" == "CUDA" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: GPUTYPE=CUDA but nvidia-smi is not in PATH." >&2
    echo "If this is an Apptainer/Singularity container, run it with --nv." >&2
  else
    nvidia-smi -L >/dev/null 2>&1 || {
      echo "FATAL: GPUTYPE=CUDA but nvidia-smi cannot see an NVIDIA GPU." >&2
      echo "If this is an Apptainer/Singularity container, run it with --nv." >&2
      exit 1
    }
  fi

  if ! ldconfig -p 2>/dev/null | grep -q 'libcuda.so.1' && \
     ! find ${LD_LIBRARY_PATH//:/ } -maxdepth 1 -name 'libcuda.so.1*' 2>/dev/null | grep -q .; then
    echo "WARNING: GPUTYPE=CUDA but libcuda.so.1 is not visible via ldconfig or LD_LIBRARY_PATH." >&2
    echo "This usually means the container was not started with --nv, or the host NVIDIA driver is not mounted." >&2
  fi
fi

# Check HIP runtime visibility before starting the full workflow.
if [[ "$GPUTYPE" == "HIP" ]]; then
  if ! ldconfig -p 2>/dev/null | grep -q 'libamdhip64.so.6' && \
     ! find ${LD_LIBRARY_PATH//:/ } -maxdepth 1 -name 'libamdhip64.so.6*' 2>/dev/null | grep -q .; then
    echo "FATAL: GPUTYPE=HIP but libamdhip64.so.6 is not visible." >&2
    echo "Current LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
    echo "Set ROCM_PATH=/opt/rocm or BENCH_EXTRA_LD_LIBRARY_PATH=/path/to/rocm/lib" >&2
    exit 1
  fi
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
echo "# NGPUS=$NGPUS GPUTYPE=$GPUTYPE"
echo "# O2_GPU_DOUBLE_PIPELINE=$O2_GPU_DOUBLE_PIPELINE O2_GPU_RTC=$O2_GPU_RTC"
echo "# NTIMEFRAMES=$NTIMEFRAMES TFLOOP=$TFLOOP"
echo "# FILEWORKDIR=${FILEWORKDIR:-} INPUT_FILE_LIST=${INPUT_FILE_LIST:-}"
echo "# LD_LIBRARY_PATH font-lib workaround: BENCH_USE_SYSTEM_FONT_LIBS=$BENCH_USE_SYSTEM_FONT_LIBS"
echo "# ROCm library auto-detect: BENCH_AUTO_ROCM_LIBS=$BENCH_AUTO_ROCM_LIBS (active only when GPUTYPE=HIP)"
echo

export WORKFLOWMODE="print"
cmdfile="$OUTDIR/workflow_${BENCH_TAG}_${BENCH_STAMP}.sh"
echo "# Generating workflow only; command file: $cmdfile"
(
  cd "$RUNDIR"
  "$O2_DPL_WORKFLOW"
) > "$cmdfile"

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  export WORKFLOWMODE="${WORKFLOWMODE:-run}"
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
  # Analyze gpu-reconstruction processing timeslice timing and write PNG next to the log.

  : "${GPU_RECO_ANALYZER:=$O2_ROOT/prodtests/full-system-test/analyze_gpu_benchmarks.py}"

  if [[ -f "$GPU_RECO_ANALYZER" ]]; then
    analysis_png="${log%.log}_gpu_reconstruction_times.png"

    echo "# Analyzing gpu-reconstruction timeslices"
    echo "# analyzer: $GPU_RECO_ANALYZER"
    echo "# plot:     $analysis_png"

    python3 "$GPU_RECO_ANALYZER" --logfile "$log" --output "$analysis_png" || {
      echo "WARNING: gpu-reconstruction timing analysis failed" >&2
    }
  else
    echo "WARNING: gpu-reconstruction analyzer not found: $GPU_RECO_ANALYZER" >&2
  fi

  exit "$status"
fi
