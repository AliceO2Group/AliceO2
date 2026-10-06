#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${1:-${SCRIPT_DIR}/net.onnx}
if [[ ! -f $MODEL ]]; then
  echo "onnxruntime-inference-test: model not found: $MODEL" >&2
  exit 2
fi

TESTER=${ONNXRUNTIME_INFERENCE_TEST_BINARY:-${SCRIPT_DIR}/onnxruntime-ep-inference}

if [[ -n ${GPU_SYSTEM_ROOT:-} && -f $GPU_SYSTEM_ROOT/etc/gpu-features-available.sh ]]; then
  source "$GPU_SYSTEM_ROOT/etc/gpu-features-available.sh"
fi

add_cuda_driver_path() {
  local roots=()
  local root candidate

  shopt -s nullglob
  roots+=(/usr/local/cuda*)
  shopt -u nullglob
  [[ -n ${O2_GPU_CUDA_HOME:-} ]] && roots+=("$O2_GPU_CUDA_HOME")
  roots+=(/usr/local/nvidia/lib64 /usr/local/nvidia/lib)
  roots+=(/usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/wsl/lib)

  for root in "${roots[@]}"; do
    [[ -d $root ]] || continue
    candidate=$(find "$root" -type d -name stubs -prune -false -o \
      \( -type f -o -type l \) \( -name libcuda.so -o -name libcuda.so.1 \) \
      -printf '%h\n' -quit 2>/dev/null || true)
    if [[ -n $candidate ]]; then
      export LD_LIBRARY_PATH="$candidate${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      echo "onnxruntime-inference-test: added CUDA driver library path: $candidate"
      return 0
    fi
  done

  echo "onnxruntime-inference-test: no CUDA driver library found; LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
}

add_cuda_driver_path

exec "$TESTER" \
  --model "$MODEL" \
  --provider cuda \
  --device-id "${ONNXRUNTIME_INFERENCE_TEST_DEVICE_ID:-0}" \
  --expected-input-elements "${ONNXRUNTIME_INFERENCE_TEST_EXPECTED_INPUTS:-246}" \
  --expected-output-elements "${ONNXRUNTIME_INFERENCE_TEST_EXPECTED_OUTPUTS:-7}"
