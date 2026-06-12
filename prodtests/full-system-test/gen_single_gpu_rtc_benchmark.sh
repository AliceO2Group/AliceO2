#!/usr/bin/env bash
# Generate or run a single-GPU reconstruction benchmark workflow using dpl-workflow.sh.
#
# Main benchmark mode:
#   BENCHMARK_RUN=1 FILEWORKDIR=/path/to/raw_tf_dir ./gen_single_gpu_rtc_benchmark.sh

set -euo pipefail

rm -rf /dev/shm/fmq*

# ----------------------------------------------------------------------------------------------------------------------
# Locate original workflow script. Keep the original untouched.

: "${GEN_TOPO_MYDIR:=$(dirname "$(realpath "$0")")}"
: "${O2_DPL_WORKFLOW:=$GEN_TOPO_MYDIR/dpl-workflow.sh}"
: "${GEN_TOPO_SOURCE_SCRIPT:=$O2_DPL_WORKFLOW}"   # backward compatibility only
O2_DPL_WORKFLOW="$GEN_TOPO_SOURCE_SCRIPT"

if [[ ! -f "$O2_DPL_WORKFLOW" ]]; then
  echo "FATAL: dpl workflow script does not exist: $O2_DPL_WORKFLOW" >&2
  echo "Set O2_DPL_WORKFLOW=/path/to/dpl-workflow.sh" >&2
  exit 1
fi

# Helper lookup must remain compatible with the original script directory.
export GEN_TOPO_MYDIR="$(dirname "$(realpath "$O2_DPL_WORKFLOW")")"

# ----------------------------------------------------------------------------------------------------------------------
# Match the normal FST startup environment.

# ----------------------------------------------------------------------------------------------------------------------
# Benchmark defaults. All can be overridden by exporting variables before calling this script.

export DPL_REPORT_PROCESSING="${DPL_REPORT_PROCESSING:-1}"

export FST_TMUX_NO_EPN="${FST_TMUX_NO_EPN:-1}"
export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS:-GPU,CTF}"
export GPUTYPE="${GPUTYPE:-CUDA}"
export NGPUS=1
export NUMAGPUIDS=1
export NUMAID="${NUMAID:-0}"

export O2_GPU_DOUBLE_PIPELINE="${O2_GPU_DOUBLE_PIPELINE:-1}"
export O2_GPU_RTC="${O2_GPU_RTC:-1}"

export EPNSYNCMODE="${EPNSYNCMODE:-0}"
export SYNCMODE="${SYNCMODE:-1}"
export SYNCRAWMODE="${SYNCRAWMODE:-0}"

export TIMEFRAME_RATE_LIMIT="${TIMEFRAME_RATE_LIMIT:-5}"
export GEN_TOPO_NO_TF_RATE_UPSCALING="${GEN_TOPO_NO_TF_RATE_UPSCALING:-1}"

export DISABLE_ROOT_OUTPUT="${DISABLE_ROOT_OUTPUT:-1}"

# Double pipeline requires zsraw input. Therefore default to raw TF input, not CTF.
export CTFINPUT="${CTFINPUT:-0}"
export RAWTFINPUT="${RAWTFINPUT:-1}"
export DIGITINPUT="${DIGITINPUT:-0}"
export EXTINPUT="${EXTINPUT:-0}"

export NTIMEFRAMES="${NTIMEFRAMES:--1}"
export TFLOOP="${TFLOOP:-100}"
export TFDELAY="${TFDELAY:-0}"

export RUN_BENCHMARK="${RUN_BENCHMARK:-0}"

if [[ -f "$PWD/local_env.sh" ]]; then
  source "$PWD/local_env.sh"
fi

export ALICE_O2_FST="${ALICE_O2_FST:-1}"

if [[ -f "$GEN_TOPO_MYDIR/setenv.sh" ]]; then
  source "$GEN_TOPO_MYDIR/setenv.sh" || {
    echo "FATAL: setenv.sh failed: $GEN_TOPO_MYDIR/setenv.sh" >&2
    exit 1
  }
else
  echo "WARNING: setenv.sh not found: $GEN_TOPO_MYDIR/setenv.sh" >&2
fi

echo "# Alien/JAliEn environment check:"
echo "#   JALIEN_TOKEN_CERT=${JALIEN_TOKEN_CERT:-}"
echo "#   JALIEN_TOKEN_KEY=${JALIEN_TOKEN_KEY:-}"
echo "#   X509_USER_PROXY=${X509_USER_PROXY:-}"
if command -v alien-token-info >/dev/null 2>&1; then
  alien-token-info || true
else
  echo "#   alien-token-info not found in PATH"
fi
echo

# ----------------------------------------------------------------------------------------------------------------------
# Recover JAliEn token environment if alien-token-info works but token env vars are missing.

if command -v alien-token-info >/dev/null 2>&1; then
  if alien-token-info >/dev/null 2>&1; then
    uid="$(id -u)"

    for cert in \
      "/tmp/jalien_token_${uid}.pem" \
      "/tmp/jalien_token_${USER}.pem" \
      "/tmp/tokencert_${uid}.pem" \
      "/tmp/tokencert_${USER}.pem"
    do
      if [[ -f "$cert" ]]; then
        export JALIEN_TOKEN_CERT="${JALIEN_TOKEN_CERT:-$cert}"
        break
      fi
    done

    for key in \
      "/tmp/jalien_token_${uid}.key" \
      "/tmp/jalien_token_${USER}.key" \
      "/tmp/tokenkey_${uid}.pem" \
      "/tmp/tokenkey_${USER}.pem"
    do
      if [[ -f "$key" ]]; then
        export JALIEN_TOKEN_KEY="${JALIEN_TOKEN_KEY:-$key}"
        break
      fi
    done

    # Some older tools only look for X509_USER_PROXY.
    if [[ -z "${X509_USER_PROXY:-}" ]]; then
      for proxy in \
        "/tmp/x509up_u${uid}" \
        "/tmp/x509up_${uid}" \
        "${JALIEN_TOKEN_CERT:-}"
      do
        if [[ -n "$proxy" && -f "$proxy" ]]; then
          export X509_USER_PROXY="$proxy"
          break
        fi
      done
    fi
  fi
fi

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

# ----------------------------------------------------------------------------------------------------------------------
# Keep accidental files out of the source/original directory.

# Let O2/core dumps land in the benchmark run directory, not in the original working directory.
export CORE_DUMP_DIR="${CORE_DUMP_DIR:-$RUNDIR}"
export O2_CORE_DUMP_DIR="${O2_CORE_DUMP_DIR:-$RUNDIR}"
export FAIRMQ_SHM_MONITOR_CONFIG="${FAIRMQ_SHM_MONITOR_CONFIG:-}"

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
if [[ "${GPUTYPE:-}" == "HIP" && "0$BENCH_AUTO_ROCM_LIBS" == "01" ]]; then
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

# Fail early for the unsupported combination instead of letting o2-gpu-reco-workflow crash later.
if [[ "${O2_GPU_DOUBLE_PIPELINE:-0}" == "1" ]]; then
  if [[ "${CTFINPUT:-0}" == "1" ]]; then
    echo "FATAL: O2_GPU_DOUBLE_PIPELINE=1 is incompatible with CTFINPUT=1 in dpl-workflow.sh." >&2
    echo "Double pipeline requires o2-gpu-reco-workflow --input-type=zsraw." >&2
    echo "Use RAWTFINPUT=1 or rawAll.cfg input, or set O2_GPU_DOUBLE_PIPELINE=0 for CTF benchmarking." >&2
    exit 1
  fi
  if [[ "${DIGITINPUT:-0}" == "1" ]]; then
    echo "FATAL: O2_GPU_DOUBLE_PIPELINE=1 is not suitable for DIGITINPUT=1 in dpl-workflow.sh." >&2
    echo "Digit input uses zsonthefly and is restricted to NTIMEFRAMES=1." >&2
    exit 1
  fi
fi

# Input checks with clearer messages.
if [[ "${RAWTFINPUT:-0}" == "1" ]]; then
  if [[ -z "${FILEWORKDIR:-}" && -z "${INPUT_FILE_LIST:-}" ]]; then
    echo "FATAL: RAWTFINPUT=1 but neither FILEWORKDIR nor INPUT_FILE_LIST is set." >&2
    echo "Set FILEWORKDIR=/path/to/raw_tf_dir or INPUT_FILE_LIST=/path/to/o2_*.tf" >&2
    exit 1
  fi
  if [[ -z "${INPUT_FILE_LIST:-}" ]] && ! ls "${FILEWORKDIR}"/o2_*.tf >/dev/null 2>&1; then
    echo "FATAL: RAWTFINPUT=1 but no raw TF file was found." >&2
    echo "Looked for: ${FILEWORKDIR}/o2_*.tf" >&2
    echo "Set FILEWORKDIR=/path/to/raw_tf_dir or INPUT_FILE_LIST=/path/to/o2_*.tf" >&2
    exit 1
  fi
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
echo "# NGPUS=$NGPUS NUMAGPUIDS=$NUMAGPUIDS NUMAID=$NUMAID GPUTYPE=$GPUTYPE"
echo "# O2_GPU_DOUBLE_PIPELINE=$O2_GPU_DOUBLE_PIPELINE O2_GPU_RTC=$O2_GPU_RTC"
echo "# CTFINPUT=$CTFINPUT RAWTFINPUT=$RAWTFINPUT DIGITINPUT=$DIGITINPUT EXTINPUT=$EXTINPUT NTIMEFRAMES=$NTIMEFRAMES TFLOOP=$TFLOOP"
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

  # {
  #   echo "tag,host,date,ngpus,numagpuids,double_pipeline,rtc,ntf,tfloop,elapsed_seconds,max_rss_kb,exit_status"
  #   awk '
  #     /Elapsed \(wall clock\) time/ {elapsed=$NF}
  #     /Maximum resident set size/ {rss=$NF}
  #     /Exit status/ {status=$NF}
  #     END {print ENVIRON["BENCH_TAG"] "," ENVIRON["HOSTNAME"] "," strftime("%FT%T%z") "," ENVIRON["NGPUS"] "," ENVIRON["NUMAGPUIDS"] "," ENVIRON["O2_GPU_DOUBLE_PIPELINE"] "," ENVIRON["O2_GPU_RTC"] "," ENVIRON["NTIMEFRAMES"] "," ENVIRON["TFLOOP"] "," elapsed "," rss "," status}' "$log"
  # } > "$OUTDIR/summary_${BENCH_TAG}_${BENCH_STAMP}.csv"

  # echo "# Summary: $OUTDIR/summary_${BENCH_TAG}_${BENCH_STAMP}.csv"
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
