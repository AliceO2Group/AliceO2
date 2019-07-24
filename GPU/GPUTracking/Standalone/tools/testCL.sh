#!/bin/bash

COMPILER=clang++
LLVM_SPIRV=llvm-spirv

#COMPILER=/usr/lib/llvm/roc-2.1.0/bin/clang++
#COMPILER=/usr/lib/llvm/9/bin/clang++

#COMPILER=/home/qon/llvm/install/bin/clang++
#LLVM_SPIRV=/home/qon/llvm/build-spirv/tools/llvm-spirv/llvm-spirv

INCLUDES="-I../. -I../Base -I../SliceTracker -I../Common -I../Merger -I../TRDTracking -I../ITS -I../dEdx -I../TPCConvert -I../TPCFastTransformation -I../DataCompression -I$HOME/alice/O2/DataFormats/Detectors/TPC/include -I$HOME/alice/O2/Detectors/Base/include -I$HOME/alice/O2/Detectors/Base/src -I$HOME/alice/O2/Common/MathUtils/include -I$HOME/alice/O2/Detectors/TRD/base/include -I$HOME/alice/O2/Detectors/TRD/base/src -I$HOME/alice/O2/Detectors/ITSMFT/ITS/tracking/include -I$HOME/alice/O2/Detectors/ITSMFT/ITS/tracking/cuda/include -I$HOME/alice/O2/Common/Constants/include"
DEFINES="-DGPUCA_STANDALONE -DGPUCA_ENABLE_GPU_TRACKER -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -DHAVE_O2HEADERS"
FLAGS="-cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -Dcl_clang_storage_class_specifiers -Wno-invalid-constexpr"

echo Test1 - Preprocess
echo $COMPILER -cl-std=c++ -x cl $INCLUDES $DEFINES -Dcl_clang_storage_class_specifiers -E ../Base/opencl/GPUReconstructionOCL.cl > test.cl
     $COMPILER -cl-std=c++ -x cl $INCLUDES $DEFINES -Dcl_clang_storage_class_specifiers -E ../Base/opencl/GPUReconstructionOCL.cl > test.cl
if [ $? != 0 ]; then exit 1; fi
    #Test 1A - Compile Preprocessed
    #$COMPILER -cl-std=c++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -c test.cl -o test.o
    #exit

echo Test2 - Clang OCL
echo clang-ocl -I/usr/lib/llvm/9/lib64/clang/9.0.0/include/opencl-c.h -cl-std=c++ -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -o test-clang-ocl.o ../Base/opencl/GPUReconstructionOCL.cl
     clang-ocl -I/usr/lib/llvm/9/lib64/clang/9.0.0/include/opencl-c.h -cl-std=c++ -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -o test-clang-ocl.o ../Base/opencl/GPUReconstructionOCL.cl
if [ $? != 0 ]; then exit 1; fi

echo Test3 - SPIR-V
echo $COMPILER -cl-std=c++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS $INCLUDES $DEFINES -c ../Base/opencl/GPUReconstructionOCL.cl -o test.bc
     $COMPILER -cl-std=c++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS $INCLUDES $DEFINES -c ../Base/opencl/GPUReconstructionOCL.cl -o test.bc
if [ $? != 0 ]; then exit 1; fi
echo $LLVM_SPIRV test.bc
     $LLVM_SPIRV test.bc
if [ $? != 0 ]; then exit 1; fi

echo Test4 - amdgcn
echo $COMPILER -cl-std=c++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -c ../Base/opencl/GPUReconstructionOCL.cl -o test.o
     $COMPILER -cl-std=c++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -c ../Base/opencl/GPUReconstructionOCL.cl -o test.o
if [ $? != 0 ]; then exit 1; fi
