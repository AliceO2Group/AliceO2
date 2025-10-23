#!/bin/bash
if [[ -z $1 ]]; then
  echo "Please provide Sourcedir as command line argument"
  exit 1
fi
set -e
mkdir -p standalone/build
pushd standalone/build
cp $1/GPU/GPUTracking/Standalone/cmake/config.cmake .
if [[ $GPUCA_STANDALONE_CI == 1 ]]; then
  cat >> config.cmake << "EOF"
  set(ENABLE_CUDA 1)
  set(ENABLE_HIP 1)
  set(ENABLE_OPENCL 1)
  set(GPUCA_CONFIG_ONNX 1)
  set(GPUCA_BUILD_EVENT_DISPLAY_QT 0)
  set(GPUCA_CONFIG_WERROR 1)
EOF
fi
cmake -DCMAKE_INSTALL_PREFIX=../ $1/GPU/GPUTracking/Standalone
make ${JOBS+-j $JOBS} install
if [[ $GPUCA_STANDALONE_CI == 1 ]]; then
  cat >> config.cmake << "EOF"
  set(ENABLE_CUDA 0)
  set(ENABLE_HIP 0)
  set(ENABLE_OPENCL 0)
  set(GPUCA_BUILD_EVENT_DISPLAY 0)
  set(GPUCA_CONFIG_ONNX 0)
  set(GPUCA_CONFIG_VC 0)
  set(GPUCA_CONFIG_FMT 0)
  set(GPUCA_CONFIG_ROOT 0)
EOF
  make ${JOBS+-j $JOBS} install
fi
popd
