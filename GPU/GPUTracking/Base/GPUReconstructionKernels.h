// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionKernels.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONKERNELS_H
#define GPURECONSTRUCTIONKERNELS_H

namespace o2
{
namespace gpu
{
#ifdef GPUCA_GPURECONSTRUCTIONCPU_DECLONLY
template <>
class GPUReconstructionKernels<GPUReconstructionCPUBackend> : public GPUReconstructionCPUBackend
#define GPUCA_KRNL(...) ;
#define GPUCA_KRNL_CLASS GPUReconstructionCPUBackend
#else
template <class T>
class GPUReconstructionKernels : public T
#define GPUCA_EXPAND(...) __VA_ARGS__
#define GPUCA_KRNL(X) GPUCA_EXPAND X
#define GPUCA_KRNL_CLASS T
#endif
{
 public:
  virtual ~GPUReconstructionKernels() = default; // Do not declare override in template class! AMD hcc will not create the destructor otherwise.
  GPUReconstructionKernels(const GPUSettingsProcessing& cfg) : GPUCA_KRNL_CLASS(cfg) {}

 protected:
  virtual int runKernelImpl(classArgument<GPUTPCNeighboursFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCNeighboursFinder>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUTPCNeighboursCleaner>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCNeighboursCleaner>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUTPCStartHitsFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCStartHitsFinder>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUTPCStartHitsSorter>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCStartHitsSorter>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUTPCTrackletConstructor>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCTrackletConstructor>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUTPCTrackletConstructor, 1>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCTrackletConstructor, 1>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUTPCTrackletSelector>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCTrackletSelector>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUMemClean16>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, void* ptr, unsigned long size) GPUCA_KRNL(({ return T::template runKernelBackend<GPUMemClean16>(x, y, z, ptr, size); }));
  virtual int runKernelImpl(classArgument<GPUTPCGMMergerTrackFit>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCGMMergerTrackFit>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUTRDTrackerGPU>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTRDTrackerGPU>(x, y, z); }));
  virtual int runKernelImpl(classArgument<GPUITSFitterKernel>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUITSFitterKernel>(x, y, z); }));
};

#undef GPUCA_KRNL
#undef GPUCA_KRNL_CLASS
}
} // namespace o2::gpu

#endif
