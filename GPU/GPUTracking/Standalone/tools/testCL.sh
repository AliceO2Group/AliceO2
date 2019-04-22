COMPILER=clang++
LLVM_SPIRV=llvm-spirv

#COMPILER=/usr/lib/llvm/roc-2.1.0/bin/clang++
#COMPILER=/usr/lib/llvm/9/bin/clang++

#COMPILER=/home/qon/llvm/install/bin/clang++
#LLVM_SPIRV=/home/qon/llvm/build-spirv/tools/llvm-spirv/llvm-spirv

echo Test1 - Preprocess
echo $COMPILER -cl-std=c++ -x cl -I ../. -I ../Base -I ../SliceTracker -I ../Common -I ../Merger -I ../TRDTracking -I../ITS -DGPUCA_STANDALONE -DGPUCA_ENABLE_GPU_TRACKER -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -Dcl_clang_storage_class_specifiers -E ../Base/opencl/GPUReconstructionOCL.cl > test.cl
     $COMPILER -cl-std=c++ -x cl -I ../. -I ../Base -I ../SliceTracker -I ../Common -I ../Merger -I ../TRDTracking -I../ITS -DGPUCA_STANDALONE -DGPUCA_ENABLE_GPU_TRACKER -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -Dcl_clang_storage_class_specifiers -E ../Base/opencl/GPUReconstructionOCL.cl > test.cl

echo Test2 - amdgcn
echo $COMPILER -cl-std=c++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -I../. -I ../Base -I ../SliceTracker -I ../Common -I ../Merger -I ../TRDTracking -I../ITS -DGPUCA_STANDALONE -DGPUCA_ENABLE_GPU_TRACKER -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -Dcl_clang_storage_class_specifiers -c ../Base/opencl/GPUReconstructionOCL.cl -o test.o
     $COMPILER -cl-std=c++ -x cl --target=amdgcn-amd-amdhsa -mcpu=gfx906 -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -I../. -I ../Base -I ../SliceTracker -I ../Common -I ../Merger -I ../TRDTracking -I../ITS -DGPUCA_STANDALONE -DGPUCA_ENABLE_GPU_TRACKER -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -Dcl_clang_storage_class_specifiers -c ../Base/opencl/GPUReconstructionOCL.cl -o test.o

echo Test3 - SPIR-V
echo $COMPILER -cl-std=c++ -x cl -emit-llvm --target=spir64-unknown-unknown -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -I../. -I ../Base -I ../SliceTracker -I ../Common -I ../Merger -I ../TRDTracking -I../ITS -DGPUCA_STANDALONE -DGPUCA_ENABLE_GPU_TRACKER -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -Dcl_clang_storage_class_specifiers -c ../Base/opencl/GPUReconstructionOCL.cl -o test.bc
     $COMPILER -cl-std=c++ -x cl -emit-llvm --target=spir64-unknown-unknown -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -I../. -I ../Base -I ../SliceTracker -I ../Common -I ../Merger -I ../TRDTracking -I../ITS -DGPUCA_STANDALONE -DGPUCA_ENABLE_GPU_TRACKER -DGPUCA_GPULIBRARY=OCL -DNDEBUG -D__OPENCLCPP__ -Dcl_clang_storage_class_specifiers -c ../Base/opencl/GPUReconstructionOCL.cl -o test.bc
echo $LLVM_SPIRV test.bc
     $LLVM_SPIRV test.bc
