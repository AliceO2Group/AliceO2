#!/bin/bash
cat <<EOT > source.cu
# 1 "/home/qon/alice/O2/GPU/GPUTracking/Base/cuda/GPUReconstructionCUDArtc.cu"
# 1 "/home/qon/alice/O2/GPU/GPUTracking/Base/cuda/GPUReconstructionCUDArtcPre.h" 1
#include <type_traits>
int main(int, char**) {return 0;}
EOT
nvcc -E \
  -I src/ -I src/Common/ -I src/Base/ -I src/SliceTracker/ -I src/Merger/ -I src/TRDTracking/ -I src/TPCClusterFinder/ -I src/TPCConvert/ -I src/Global/ -I src/dEdx/ -I src/TPCFastTransformation/ -I src/GPUUtils/ -I src/DataCompression -I src/ITS \
  -I$HOME/alice/O2/DataFormats/Detectors/TPC/include -I$HOME/alice/O2/Detectors/Base/include -I$HOME/alice/O2/Detectors/Base/src -I$HOME/alice/O2/Common/MathUtils/include -I$HOME/alice/O2/DataFormats/Headers/include \
  -I$HOME/alice/O2/Detectors/TRD/base/include -I$HOME/alice/O2/Detectors/TRD/base/src -I$HOME/alice/O2/Detectors/ITSMFT/ITS/tracking/include -I$HOME/alice/O2/Detectors/ITSMFT/ITS/tracking/cuda/include -I$HOME/alice/O2/Common/Constants/include \
  -I$HOME/alice/O2/DataFormats/common/include -I$HOME/alice/O2/DataFormats/Detectors/TRD/include -I$HOME/alice/O2/Detectors/Raw/include \
  -DHAVE_O2HEADERS -DGPUCA_TPC_GEOMETRY_O2 -DGPUCA_STANDALONE -DGPUCA_NO_ITS_TRAITS -DGPUCA_GPUCODE_GENRTC -D__CUDA_ARCH__=750 \
  ~/alice/O2/GPU/GPUTracking/Base/cuda/GPUReconstructionCUDArtc.cu \
| sed '1,/^# 1 ".*GPUReconstructionCUDArtcPre.h" 1$/d' \
>> source.cu
