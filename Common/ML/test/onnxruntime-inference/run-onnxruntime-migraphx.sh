#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${1:-${SCRIPT_DIR}/net.onnx}
if [[ ! -f $MODEL ]]; then
  echo "onnxruntime-inference-test: model not found: $MODEL" >&2
  exit 2
fi

TESTER=${ONNXRUNTIME_INFERENCE_TEST_BINARY:-${SCRIPT_DIR}/onnxruntime-ep-inference}

exec "$TESTER" \
  --model "$MODEL" \
  --provider migraphx \
  --device-id "${ONNXRUNTIME_INFERENCE_TEST_DEVICE_ID:-0}" \
  --expected-input-elements "${ONNXRUNTIME_INFERENCE_TEST_EXPECTED_INPUTS:-246}" \
  --expected-output-elements "${ONNXRUNTIME_INFERENCE_TEST_EXPECTED_OUTPUTS:-7}"
