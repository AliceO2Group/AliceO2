#!/bin/bash

COMPILER=clang++
LLVM_SPIRV=llvm-spirv

#COMPILER=/usr/lib/llvm/roc-2.1.0/bin/clang++
#COMPILER=/usr/lib/llvm/9/bin/clang++

#COMPILER=/home/qon/clang-build/install/bin/clang++
#LLVM_SPIRV=/home/qon/clang-build/SPIRV-LLVM-Translator/build/tools/llvm-spirv/llvm-spirv

INCLUDES="-I../. -I../DataTypes -I../Definitions -I../Base -I../SliceTracker -I../Common -I../Merger -I../Refit -I../TRDTracking -I../ITS -I../dEdx -I../TPCConvert -I../TPCFastTransformation -I../DataCompression -I../TPCClusterFinder -I../Global -I ../GPUUtils \
          -I$HOME/alice/O2/DataFormats/Detectors/TPC/include -I$HOME/alice/O2/Detectors/Base/include -I$HOME/alice/O2/Detectors/Base/src -I$HOME/alice/O2/Common/MathUtils/include -I$HOME/alice/O2/DataFormats/Headers/include \
          -I$HOME/alice/O2/Detectors/TRD/base/include -I$HOME/alice/O2/Detectors/TRD/base/src -I$HOME/alice/O2/Detectors/ITSMFT/ITS/tracking/include -I$HOME/alice/O2/Detectors/ITSMFT/ITS/tracking/cuda/include -I$HOME/alice/O2/Common/Constants/include \
          -I$HOME/alice/O2/DataFormats/common/include -I$HOME/alice/O2/DataFormats/Detectors/TRD/include -I$HOME/alice/O2/DataFormats/Reconstruction/include -I$HOME/alice/O2/DataFormats/Reconstruction/src"
DEFINES="-DGPUCA_STANDALONE -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -DHAVE_O2HEADERS -DGPUCA_TPC_GEOMETRY_O2"
FLAGS="-Xclang -fdenormal-fp-math-f32=ieee -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -Dcl_clang_storage_class_specifiers"

echo Test1 - Preprocess
echo $COMPILER -cl-std=clc++ -x cl $INCLUDES $DEFINES -Dcl_clang_storage_class_specifiers -E ../Base/opencl-common/GPUReconstructionOCL.cl > test.cl
     $COMPILER -cl-std=clc++ -x cl $INCLUDES $DEFINES -Dcl_clang_storage_class_specifiers -E ../Base/opencl-common/GPUReconstructionOCL.cl > test.cl
if [ $? != 0 ]; then exit 1; fi
#Test 1A - Compile Preprocessed
#echo $COMPILER -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS -c test.cl -o test.bc
#     $COMPILER -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS -c test.cl -o test.bc
#exit

echo
echo Test2 - Clang OCL
echo clang-ocl -O3 -cl-std=clc++ -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -o test-clang-ocl.o ../Base/opencl-common/GPUReconstructionOCL.cl
     clang-ocl -O3 -cl-std=clc++ -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -o test-clang-ocl.o ../Base/opencl-common/GPUReconstructionOCL.cl
rm -f test-clang-ocl.o.*
if [ $? != 0 ]; then exit 1; fi

echo
echo Test3 - SPIR-V
echo $COMPILER -O0 -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS $INCLUDES $DEFINES -c ../Base/opencl-common/GPUReconstructionOCL.cl -o test.bc
     $COMPILER -O0 -cl-std=clc++ -x cl -emit-llvm --target=spir64-unknown-unknown $FLAGS $INCLUDES $DEFINES -c ../Base/opencl-common/GPUReconstructionOCL.cl -o test.bc
if [ $? != 0 ]; then exit 1; fi
echo $LLVM_SPIRV test.bc -o test.spirv
     $LLVM_SPIRV test.bc -o test.spirv
if [ $? != 0 ]; then exit 1; fi

echo
echo Test4 - amdgcn
echo $COMPILER -O3 -cl-std=clc++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -c ../Base/opencl-common/GPUReconstructionOCL.cl -o test.o
     $COMPILER -O3 -cl-std=clc++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 $FLAGS $INCLUDES $DEFINES -c ../Base/opencl-common/GPUReconstructionOCL.cl -o test.o
if [ $? != 0 ]; then exit 1; fi
