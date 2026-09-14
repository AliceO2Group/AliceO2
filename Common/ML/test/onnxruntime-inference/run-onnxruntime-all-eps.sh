#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${1:-${SCRIPT_DIR}/net.onnx}
if [[ ! -f $MODEL ]]; then
  echo "onnxruntime-inference-test: model not found: $MODEL" >&2
  exit 2
fi

if [[ -n ${ONNXRUNTIME_ROOT:-} && -f $ONNXRUNTIME_ROOT/etc/ort-init.sh ]]; then
  source "$ONNXRUNTIME_ROOT/etc/ort-init.sh"
fi

has_cuda_device() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
  fi
  compgen -G "/proc/driver/nvidia/gpus/*" >/dev/null
}

has_rocm_device() {
  if command -v rocm-smi >/dev/null 2>&1 && rocm-smi -i >/dev/null 2>&1; then
    return 0
  fi
  [[ -e /dev/kfd ]] && compgen -G "/dev/dri/renderD*" >/dev/null
}

if [[ -n ${ONNXRUNTIME_INFERENCE_TEST_PROVIDERS:-} ]]; then
  IFS=', ' read -r -a PROVIDERS <<< "$ONNXRUNTIME_INFERENCE_TEST_PROVIDERS"
else
  PROVIDERS=(cpu)
  if [[ ${ORT_MIGRAPHX_BUILD:-0} == 1 ]]; then
    if has_rocm_device; then
      PROVIDERS+=(migraphx)
    else
      echo "onnxruntime-inference-test: skipping migraphx, no ROCm device detected"
    fi
  fi
  if [[ ${ORT_CUDA_BUILD:-0} == 1 ]]; then
    if has_cuda_device; then
      PROVIDERS+=(cuda)
    else
      echo "onnxruntime-inference-test: skipping cuda, no CUDA device detected"
    fi
  fi
  if [[ ${ORT_TENSORRT_BUILD:-0} == 1 ]]; then
    if has_cuda_device; then
      PROVIDERS+=(tensorrt)
    else
      echo "onnxruntime-inference-test: skipping tensorrt, no CUDA device detected"
    fi
  fi
fi

echo "onnxruntime-inference-test: selected providers: ${PROVIDERS[*]}"

FAILURES=()
for PROVIDER in "${PROVIDERS[@]}"; do
  [[ -n $PROVIDER ]] || continue
  PROVIDER=${PROVIDER,,}
  echo "onnxruntime-inference-test: running ${PROVIDER}"
  if ! "${SCRIPT_DIR}/run-onnxruntime-${PROVIDER}.sh" "$MODEL"; then
    FAILURES+=("$PROVIDER")
  fi
done

if [[ ${#FAILURES[@]} != 0 ]]; then
  echo "onnxruntime-inference-test: failed providers: ${FAILURES[*]}" >&2
  exit 1
fi

echo "onnxruntime-inference-test: all providers passed"
