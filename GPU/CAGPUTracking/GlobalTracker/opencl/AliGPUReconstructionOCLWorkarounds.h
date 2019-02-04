#ifndef ALIGPURECONSTRUCTIONOCLWORKAROUNDS_H
#define ALIGPURECONSTRUCTIONOCLWORKAROUNDS_H

//Need one workaround per OCL kernel that is used: Must fetch the ocl kernel object depending on template parameters

template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCNeighboursFinder>() {return mInternals->kernel_neighbours_finder;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCNeighboursCleaner>() {return mInternals->kernel_neighbours_cleaner;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCStartHitsFinder>() {return mInternals->kernel_start_hits_finder;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCStartHitsSorter>() {return mInternals->kernel_start_hits_sorter;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletConstructor>() {return mInternals->kernel_tracklet_constructor0;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletConstructor, 1>() {return mInternals->kernel_tracklet_constructor1;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUTPCTrackletSelector>() {return mInternals->kernel_tracklet_selector;}
template <> cl_kernel& AliGPUReconstructionOCLBackend::getKernelObject<cl_kernel, AliGPUMemClean16>() {return mInternals->kernel_memclean16;}

#endif
