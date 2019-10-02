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

namespace GPUCA_NAMESPACE
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
  using krnlRunRange = GPUReconstruction::krnlRunRange;
  using krnlExec = GPUReconstruction::krnlExec;
  using krnlEvent = GPUReconstruction::krnlEvent;
  template <class X, int Y = 0>
  using classArgument = GPUReconstruction::classArgument<X, Y>;

  virtual ~GPUReconstructionKernels() = default; // NOLINT: Do not declare override in template class! AMD hcc will not create the destructor otherwise.
  GPUReconstructionKernels(const GPUSettingsProcessing& cfg) : GPUCA_KRNL_CLASS(cfg) {}

 protected:
  virtual int runKernelImpl(classArgument<GPUTPCNeighboursFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCNeighboursFinder>(x, y, z); }));                         // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCNeighboursCleaner>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCNeighboursCleaner>(x, y, z); }));                       // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCStartHitsFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCStartHitsFinder>(x, y, z); }));                           // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCStartHitsSorter>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCStartHitsSorter>(x, y, z); }));                           // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCTrackletConstructor>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCTrackletConstructor>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCTrackletConstructor, 1>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCTrackletConstructor, 1>(x, y, z); }));             // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCTrackletSelector>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCTrackletSelector>(x, y, z); }));                         // NOLINT
  virtual int runKernelImpl(classArgument<GPUMemClean16>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, void* ptr, unsigned long size) GPUCA_KRNL(({ return T::template runKernelBackend<GPUMemClean16>(x, y, z, ptr, size); })); // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCGMMergerTrackFit>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCGMMergerTrackFit>(x, y, z); }));                         // NOLINT
#ifdef HAVE_O2HEADERS
  virtual int runKernelImpl(classArgument<GPUTRDTrackerGPU>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTRDTrackerGPU>(x, y, z); }));                                             // NOLINT
  virtual int runKernelImpl(classArgument<GPUITSFitterKernel>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUITSFitterKernel>(x, y, z); }));                                         // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCConvertKernel>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCConvertKernel>(x, y, z); }));                                       // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCCompressionKernels, 0>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCCompressionKernels, 0>(x, y, z); }));                       // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCCompressionKernels, 1>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCCompressionKernels, 1>(x, y, z); }));                       // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 0>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 0>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 1>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 1>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 2>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 2>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 3>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 3>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 4>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 4>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 5>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 5>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 6>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 6>(x, y, z); }));                   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 7>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, int iBuf) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 7>(x, y, z, iBuf); }));   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 8>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, int iBuf) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 8>(x, y, z, iBuf); }));   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 9>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, int iBuf) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 9>(x, y, z, iBuf); }));   // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 10>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, int iBuf) GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 10>(x, y, z, iBuf); })); // NOLINT
  virtual int runKernelImpl(classArgument<GPUTPCClusterFinderKernels, 11>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, int iBuf, gpucf::PackedDigit* in, gpucf::PackedDigit* out)
    GPUCA_KRNL(({ return T::template runKernelBackend<GPUTPCClusterFinderKernels, 11>(x, y, z, iBuf, in, out); })); // NOLINT
#endif
};

#undef GPUCA_KRNL
#undef GPUCA_KRNL_CLASS
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
