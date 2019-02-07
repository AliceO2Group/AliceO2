#ifndef ALIGPURECONSTRUCTIONOCLWORKAROUNDS_H
#define ALIGPURECONSTRUCTIONOCLWORKAROUNDS_H

//Need one workaround per OCL kernel that is used: Must fetch the ocl kernel object depending on template parameters
static cl_kernel krnlNULL = 0;
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCNeighboursFinder>(int num) {return num == 0 ? mInternals->kernel_neighbours_finder : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCNeighboursCleaner>(int num) {return num == 0 ? mInternals->kernel_neighbours_cleaner : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCStartHitsFinder>(int num) {return num == 0 ? mInternals->kernel_start_hits_finder : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCStartHitsSorter>(int num) {return num == 0 ? mInternals->kernel_start_hits_sorter : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletConstructor>(int num) {return num == 0 ? mInternals->kernel_tracklet_constructor0 : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletConstructor, 1>(int num) {return num == -1 ? mInternals->kernel_tracklet_constructor1 : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletSelector>(int num) {return num > 0 ? mInternals->kernel_tracklet_selector : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUMemClean16>(int num) {return num == -1 ? mInternals->kernel_memclean16 : krnlNULL;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCGMMergerTrackFit>(int num) {throw::std::runtime_error("OpenCL Merger not supported"); return krnlNULL;}

#endif
