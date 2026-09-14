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

if [[ -n ${ONNXRUNTIME_INFERENCE_TEST_PROVIDERS:-} ]]; then
  IFS=', ' read -r -a PROVIDERS <<< "$ONNXRUNTIME_INFERENCE_TEST_PROVIDERS"
else
  PROVIDERS=(cpu)
  [[ ${ORT_MIGRAPHX_BUILD:-0} == 1 ]] && PROVIDERS+=(migraphx)
  [[ ${ORT_CUDA_BUILD:-0} == 1 ]] && PROVIDERS+=(cuda)
  [[ ${ORT_TENSORRT_BUILD:-0} == 1 ]] && PROVIDERS+=(tensorrt)
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
