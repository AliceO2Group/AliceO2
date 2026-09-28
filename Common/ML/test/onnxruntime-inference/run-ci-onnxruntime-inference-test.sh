#!/usr/bin/env bash
set -euo pipefail

# CI entry point for the ONNX Runtime execution-provider inference test.
#
# This is meant to be called from an alidist recipe (O2-GPU-test) that has the
# O2 dependency environment loaded (ONNXRuntime, gpu-system). It
# derives the GPU backends to test the same way the other O2 GPU CI recipes do
# (O2GPUCI_BACKENDS or gpu-features-available.sh), maps them onto the ONNX
# Runtime execution providers ONNXRuntime was built with (ort-init.sh), and
# fails if no GPU execution provider ends up being tested: a GPU CI check that
# silently tests only the CPU provider is not a GPU CI check.
#
# usage: run-ci-onnxruntime-inference-test.sh
#   Runs the executable and model installed alongside this script.
#
# Environment:
#   O2GPUCI_BACKENDS   Comma/space separated list of backends to require
#                      (CUDA, HIP). Defaults to what gpu-system detected.
#   ONNXRUNTIME_INFERENCE_TEST_BINARY   Optional test executable override.
#   ONNXRUNTIME_INFERENCE_TEST_DEVICE_ID   GPU device id, defaults to 0.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ONNXRUNTIME_INFERENCE_TEST_BINARY=${ONNXRUNTIME_INFERENCE_TEST_BINARY:-$SCRIPT_DIR/o2-test-ml-onnxruntime-ep-inference}

if [[ -n ${ONNXRUNTIME_ROOT:-} && -f $ONNXRUNTIME_ROOT/etc/ort-init.sh ]]; then
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

export ONNXRUNTIME_INFERENCE_TEST_PROVIDERS=$(IFS=,; echo "${PROVIDERS[*]}")
echo "onnxruntime-inference-ci: backends: ${GPU_BACKENDS[*]}; providers: $ONNXRUNTIME_INFERENCE_TEST_PROVIDERS"

exec "$SCRIPT_DIR/run-onnxruntime-all-eps.sh" "$SCRIPT_DIR/net.onnx"
