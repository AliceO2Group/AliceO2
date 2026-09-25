#!/usr/bin/env bash
set -euo pipefail

# CI entry point for the ONNX Runtime execution-provider inference test.
#
# This is meant to be called from an alidist recipe (O2-GPU-test) that has the
# O2 dependency environment loaded (ONNXRuntime, gpu-system, CMake, ninja). It
# derives the GPU backends to test the same way the other O2 GPU CI recipes do
# (O2GPUCI_BACKENDS or gpu-features-available.sh), maps them onto the ONNX
# Runtime execution providers ONNXRuntime was built with (ort-init.sh), and
# fails if no GPU execution provider ends up being tested: a GPU CI check that
# silently tests only the CPU provider is not a GPU CI check.
#
# usage: run-ci-onnxruntime-inference-test.sh [O2_SOURCEDIR]
#   O2_SOURCEDIR   AliceO2 source tree. Defaults to the tree this script is in.
#
# Environment:
#   O2GPUCI_BACKENDS   Comma/space separated list of backends to require
#                      (CUDA, HIP). Defaults to what gpu-system detected.
#   BUILDDIR           If set (as in alidist recipes), the test is built in
#                      $BUILDDIR/onnxruntime-inference-test.
#   ONNXRUNTIME_INFERENCE_TEST_DEVICE_ID   GPU device id, defaults to 0.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
O2_SOURCEDIR=${1:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
TEST_DIR=$O2_SOURCEDIR/Common/ML/test/onnxruntime-inference
TEST_SCRIPT=$TEST_DIR/run-local-onnxruntime-inference-test.sh

if [[ ! -f $TEST_SCRIPT ]]; then
  echo "onnxruntime-inference-ci: could not find test runner: $TEST_SCRIPT" >&2
  echo "Pass the AliceO2 source tree as the first argument." >&2
  exit 1
fi

if [[ -z ${ONNXRUNTIME_ROOT:-} || ! -d $ONNXRUNTIME_ROOT/lib/cmake/onnxruntime ]]; then
  echo "onnxruntime-inference-ci: ONNXRUNTIME_ROOT is not set or does not contain lib/cmake/onnxruntime" >&2
  echo "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT:-}" >&2
  exit 1
fi
for TOOL in cmake ninja; do
  if ! command -v $TOOL > /dev/null; then
    echo "onnxruntime-inference-ci: $TOOL not found in PATH; add CMake and ninja to build_requires" >&2
    exit 1
  fi
done

if [[ -f $ONNXRUNTIME_ROOT/etc/ort-init.sh ]]; then
  source "$ONNXRUNTIME_ROOT/etc/ort-init.sh"
fi
if [[ -n ${GPU_SYSTEM_ROOT:-} && -f $GPU_SYSTEM_ROOT/etc/gpu-features-available.sh ]]; then
  source "$GPU_SYSTEM_ROOT/etc/gpu-features-available.sh"
fi

GPU_BACKENDS=()
if [[ -n ${O2GPUCI_BACKENDS:-} ]]; then
  read -r -a GPU_BACKENDS <<< "${O2GPUCI_BACKENDS//,/ }"
else
  [[ ${O2_GPU_CUDA_AVAILABLE:-0} == 1 ]] && GPU_BACKENDS+=(CUDA)
  [[ ${O2_GPU_ROCM_AVAILABLE:-0} == 1 ]] && GPU_BACKENDS+=(HIP)
fi

if [[ ${#GPU_BACKENDS[@]} == 0 ]]; then
  echo "onnxruntime-inference-ci: no GPU backend selected or detected." >&2
  echo "Set O2GPUCI_BACKENDS='CUDA,HIP' in CI to require both production GPU backends." >&2
  exit 1
fi

PROVIDERS=(cpu)
for BACKEND in "${GPU_BACKENDS[@]}"; do
  case "${BACKEND^^}" in
    CUDA)
      if [[ ${ORT_CUDA_BUILD:-0} != 1 ]]; then
        echo "onnxruntime-inference-ci: CUDA backend requested but ONNXRuntime was built without the CUDA execution provider" >&2
        exit 1
      fi
      PROVIDERS+=(cuda)
      [[ ${ORT_TENSORRT_BUILD:-0} == 1 ]] && PROVIDERS+=(tensorrt)
      ;;
    HIP|ROCM)
      if [[ ${ORT_MIGRAPHX_BUILD:-0} != 1 ]]; then
        echo "onnxruntime-inference-ci: HIP backend requested but ONNXRuntime was built without the MIGraphX execution provider" >&2
        exit 1
      fi
      PROVIDERS+=(migraphx)
      ;;
    *)
      echo "onnxruntime-inference-ci: unsupported backend requested: $BACKEND" >&2
      exit 1
      ;;
  esac
done

if [[ ${#PROVIDERS[@]} == 1 ]]; then
  echo "onnxruntime-inference-ci: no ONNXRuntime GPU execution provider was selected." >&2
  echo "ORT_CUDA_BUILD=${ORT_CUDA_BUILD:-0}, ORT_TENSORRT_BUILD=${ORT_TENSORRT_BUILD:-0}, ORT_MIGRAPHX_BUILD=${ORT_MIGRAPHX_BUILD:-0}" >&2
  exit 1
fi

BUILD_DIR=${BUILDDIR:-${TMPDIR:-/tmp}}/onnxruntime-inference-test
PROVIDER_LIST=$(IFS=,; echo "${PROVIDERS[*]}")
echo "onnxruntime-inference-ci: backends: ${GPU_BACKENDS[*]}; providers: $PROVIDER_LIST"

EXTRA_ARGS=()
[[ -n ${ONNXRUNTIME_INFERENCE_TEST_DEVICE_ID:-} ]] && EXTRA_ARGS+=(--device-id "$ONNXRUNTIME_INFERENCE_TEST_DEVICE_ID")

# The local runner must never fall back to alienv bootstrapping in CI.
export ONNXRUNTIME_INFERENCE_TEST_BOOTSTRAPPED=1
"$TEST_SCRIPT" --model "$TEST_DIR/net.onnx" \
               --build-dir "$BUILD_DIR" \
               --providers "$PROVIDER_LIST" \
               ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
rm -Rf "$BUILD_DIR"
