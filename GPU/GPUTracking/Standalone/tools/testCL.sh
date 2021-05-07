#!/bin/bash

COMPILER=clang++
LLVM_SPIRV=llvm-spirv

echo "Testing using clang `which clang++`, spirv `which llvm-spirv`"

#COMPILER=/usr/lib/llvm/roc-2.1.0/bin/clang++
#COMPILER=/usr/lib/llvm/9/bin/clang++

#COMPILER=/home/qon/alice/llvm-project/build/bin/clang++
#LLVM_SPIRV=/home/qon/alice/llvm-project/build/bin/llvm-spirv

O2_DIR=${HOME}/alice/O2
GPU_DIR=${HOME}/alice/O2/GPU/GPUTracking

INCLUDES="-I${GPU_DIR}/. -I${GPU_DIR}/DataTypes -I${GPU_DIR}/Definitions -I${GPU_DIR}/Base -I${GPU_DIR}/SliceTracker -I${O2_DIR}/GPU/Common -I${GPU_DIR}/Merger -I${GPU_DIR}/Refit -I${GPU_DIR}/TRDTracking -I${GPU_DIR}/ITS -I${GPU_DIR}/dEdx \
          -I${GPU_DIR}/TPCConvert -I${O2_DIR}/GPU/TPCFastTransformation -I${GPU_DIR}/DataCompression -I${GPU_DIR}/TPCClusterFinder -I${GPU_DIR}/Global -I ${O2_DIR}/GPU/Utils \
          -I${O2_DIR}/DataFormats/Detectors/TPC/include -I${O2_DIR}/Detectors/Base/include -I${O2_DIR}/Detectors/Base/src -I${O2_DIR}/Common/MathUtils/include -I${O2_DIR}/DataFormats/Headers/include \
          -I${O2_DIR}/Detectors/TRD/base/include -I${O2_DIR}/Detectors/TRD/base/src -I${O2_DIR}/Detectors/ITSMFT/ITS/tracking/include -I${O2_DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include -I${O2_DIR}/Common/Constants/include \
          -I${O2_DIR}/DataFormats/common/include -I${O2_DIR}/DataFormats/Detectors/Common/include -I${O2_DIR}/DataFormats/Detectors/TRD/include -I${O2_DIR}/DataFormats/Reconstruction/include -I${O2_DIR}/DataFormats/Reconstruction/src \
          -I${O2_DIR}/Detectors/Raw/include"
DEFINES="-DGPUCA_STANDALONE -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -DHAVE_O2HEADERS -DGPUCA_TPC_GEOMETRY_O2"
FLAGS="-Xclang -fdenormal-fp-math-f32=ieee -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Dcl_clang_storage_class_specifiers"

echo Test1 - Preprocess
echo $COMPILER -cl-std=clc++ -x cl $INCLUDES $DEFINES -Dcl_clang_storage_class_specifiers -cl-no-stdinc -E ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl > test.cl
     $COMPILER -cl-std=clc++ -x cl $INCLUDES $DEFINES -Dcl_clang_storage_class_specifiers -cl-no-stdinc -E ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl > test.cl
if [ $? != 0 ]; then exit 1; fi
echo Test 1A - Compile Preprocessed
echo $COMPILER -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS -c test.cl -o test.bc
     $COMPILER -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS -c test.cl -o test.bc

echo
echo Test2 - SPIR-V
echo $COMPILER -O0 -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS $INCLUDES $DEFINES -c ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl -o test.bc
     $COMPILER -O0 -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS $INCLUDES $DEFINES -c ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl -o test.bc
if [ $? != 0 ]; then exit 1; fi
echo $LLVM_SPIRV test.bc -o test.spirv
     $LLVM_SPIRV test.bc -o test.spirv
if [ $? != 0 ]; then exit 1; fi

echo
echo Test3 - amdgcn
echo $COMPILER -O3 -cl-std=clc++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -c ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl -o test.o
     $COMPILER -O3 -cl-std=clc++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -c ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl -o test.o
if [ $? != 0 ]; then exit 1; fi

echo
echo Test4 - Clang OCL
echo clang-ocl -O3 -cl-std=clc++ -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -o test-clang-ocl.o ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl
     clang-ocl -O3 -cl-std=clc++ -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -o test-clang-ocl.o ${GPU_DIR}/Base/opencl-common/GPUReconstructionOCL.cl
rm -f test-clang-ocl.o.*
if [ $? != 0 ]; then exit 1; fi

