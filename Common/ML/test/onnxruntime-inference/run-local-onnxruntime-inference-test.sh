#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: run-local-onnxruntime-inference-test.sh [options]

Build and run the ONNX Runtime execution-provider inference smoke test in the
currently loaded aliBuild environment. If needed, the script re-runs itself in
an environment that provides ONNXRuntime, CMake, and Ninja.

Options:
  --model FILE        ONNX model to test. Defaults to the bundled net.onnx.
  --build-dir DIR     Temporary CMake build dir. Defaults to /tmp.
  --providers LIST    Comma-separated providers to force, e.g. cpu,cuda.
                      By default, providers are selected from ort-init.sh.
  --device-id N       GPU device id passed to CUDA/MIGraphX/TensorRT tests.
  --help              Show this message.
EOF
}

ORIGINAL_ARGS=("$@")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f ${SCRIPT_DIR}/net.onnx ]]; then
  MODEL=${SCRIPT_DIR}/net.onnx
else
  MODEL=${SCRIPT_DIR}/../test/onnxruntime-inference/net.onnx
fi
BUILD_DIR=${TMPDIR:-/tmp}/onnxruntime-inference-test-local-${USER:-user}
PROVIDERS=
DEVICE_ID=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL=$2
      shift 2
      ;;
    --build-dir)
      BUILD_DIR=$2
      shift 2
      ;;
    --providers)
      PROVIDERS=$2
      shift 2
      ;;
    --device-id)
      DEVICE_ID=$2
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "run-local-onnxruntime-inference-test: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f $MODEL ]]; then
  echo "run-local-onnxruntime-inference-test: model not found: $MODEL" >&2
  exit 2
fi

detect_onnxruntime_root() {
  [[ -n ${ONNXRUNTIME_ROOT:-} && -d $ONNXRUNTIME_ROOT/lib/cmake/onnxruntime ]] && return 0

  IFS=: read -r -a SEARCH_PATHS <<< "${CMAKE_PREFIX_PATH:-}:${LD_LIBRARY_PATH:-}:${ROOT_INCLUDE_PATH:-}"
  for PATH_ENTRY in "${SEARCH_PATHS[@]}"; do
    CANDIDATE=
    case "$PATH_ENTRY" in
      */ONNXRuntime/*/lib)
        CANDIDATE=${PATH_ENTRY%/lib}
        ;;
      */ONNXRuntime/*/include/onnxruntime)
        CANDIDATE=${PATH_ENTRY%/include/onnxruntime}
        ;;
      */ONNXRuntime/*)
        CANDIDATE=$PATH_ENTRY
        ;;
    esac
    if [[ -n $CANDIDATE && -d $CANDIDATE/lib/cmake/onnxruntime ]]; then
      export ONNXRUNTIME_ROOT=$CANDIDATE
      return 0
    fi
  done
  return 1
}

if ! detect_onnxruntime_root || ! command -v cmake > /dev/null || ! command -v ninja > /dev/null; then
  if [[ ${ONNXRUNTIME_INFERENCE_TEST_BOOTSTRAPPED:-0} != 1 ]] && command -v alienv > /dev/null; then
    export ONNXRUNTIME_INFERENCE_TEST_BOOTSTRAPPED=1
    exec alienv setenv ONNXRuntime/latest,CMake/latest,ninja/latest \
      -c "$SCRIPT_DIR/run-local-onnxruntime-inference-test.sh" "${ORIGINAL_ARGS[@]}"
  fi
fi

if [[ -z ${ONNXRUNTIME_ROOT:-} ]]; then
  echo "run-local-onnxruntime-inference-test: ONNXRUNTIME_ROOT is not set" >&2
  echo "Could not infer it from the loaded environment." >&2
  exit 2
fi

if [[ -f $ONNXRUNTIME_ROOT/etc/ort-init.sh ]]; then
  source "$ONNXRUNTIME_ROOT/etc/ort-init.sh"
fi

if [[ -n $PROVIDERS ]]; then
  export ONNXRUNTIME_INFERENCE_TEST_PROVIDERS=$PROVIDERS
fi
if [[ -n $DEVICE_ID ]]; then
  export ONNXRUNTIME_INFERENCE_TEST_DEVICE_ID=$DEVICE_ID
fi

if [[ -n ${ONNXRUNTIME_INFERENCE_TEST_BINARY:-} ]]; then
  :
elif [[ -x ${SCRIPT_DIR}/onnxruntime-ep-inference && ! -f ${SCRIPT_DIR}/CMakeLists.txt ]]; then
  export ONNXRUNTIME_INFERENCE_TEST_BINARY="$SCRIPT_DIR/onnxruntime-ep-inference"
else
  rm -Rf "$BUILD_DIR"
  cmake -S "$SCRIPT_DIR" \
        -B "$BUILD_DIR" \
        -G Ninja \
        -Donnxruntime_DIR="$ONNXRUNTIME_ROOT/lib/cmake/onnxruntime" \
        -DORT_CUDA_BUILD="${ORT_CUDA_BUILD:-0}" \
        -DORT_MIGRAPHX_BUILD="${ORT_MIGRAPHX_BUILD:-0}" \
        -DORT_TENSORRT_BUILD="${ORT_TENSORRT_BUILD:-0}"
  cmake --build "$BUILD_DIR"
  export ONNXRUNTIME_INFERENCE_TEST_BINARY="$BUILD_DIR/onnxruntime-ep-inference"
fi
"$SCRIPT_DIR/run-onnxruntime-all-eps.sh" "$MODEL"
