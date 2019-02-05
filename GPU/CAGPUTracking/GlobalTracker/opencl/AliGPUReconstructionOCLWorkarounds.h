#ifndef ALIGPURECONSTRUCTIONOCLWORKAROUNDS_H
#define ALIGPURECONSTRUCTIONOCLWORKAROUNDS_H

//Need one workaround per OCL kernel that is used: Must fetch the ocl kernel object depending on template parameters
static cl_kernel krnlNo = 0;
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCNeighboursFinder>(int num) {return num == 0 ? mInternals->kernel_neighbours_finder : krnlNo;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCNeighboursCleaner>(int num) {return num == 0 ? mInternals->kernel_neighbours_cleaner : krnlNo;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCStartHitsFinder>(int num) {return num == 0 ? mInternals->kernel_start_hits_finder : krnlNo;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCStartHitsSorter>(int num) {return num == 0 ? mInternals->kernel_start_hits_sorter : krnlNo;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletConstructor>(int num) {return num == 0 ? mInternals->kernel_tracklet_constructor0 : krnlNo;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletConstructor, 1>(int num) {return num == -1 ? mInternals->kernel_tracklet_constructor1 : krnlNo;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletSelector>(int num) {return num > 0 ? mInternals->kernel_tracklet_selector : krnlNo;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUMemClean16>(int num) {return num == -1 ? mInternals->kernel_memclean16 : krnlNo;}

#endif
