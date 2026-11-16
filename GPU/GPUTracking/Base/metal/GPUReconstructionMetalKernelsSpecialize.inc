// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionMetalKernelsSpecialize.inc

#include "GPUTPCGMMergerSortComp.h"
#include <tbb/parallel_for.h>
#include <chrono>

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  const uint64_t offset = static_cast<const uint8_t*>(ptr) - static_cast<const uint8_t*>(mDeviceMemoryBase);
  const uint64_t length = (size + 15ull) & ~15ull;
  if (_xyz.z.evList) {
    StreamWaitForEvents(_xyz.x.stream, _xyz.z.evList, _xyz.z.nEvents);
  }

  id<MTLCommandBuffer> cb = mInternals->commandBuffers[_xyz.x.stream];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  metalBeginEncoder(mInternals, blit, _xyz.x.stream);

  [blit fillBuffer:mInternals->mem_gpu range:NSMakeRange(offset, length) value:0];
  metalEndEncoder(mInternals, blit, _xyz.x.stream);
  if (_xyz.z.ev) {
    RecordMarker(_xyz.z.ev, _xyz.x.stream);
  }
}

// The sorts that CUDA and HIP run with thrust (GPUCA_SPECIALIZE_THRUST_SORTS)
// run on a single GPU thread otherwise. The memory is shared with the host, so
// sort there, with the same std::sort as the CPU backend, which also orders
// the elements that compare equal as on the CPU. The number of elements is
// read once the stream is drained, from the device copy of the counters,
// since the host copy may still be on its way.
template <class T>
inline T* GPUReconstructionMetal::HostPointer(T* p)
{
  const size_t offset = (const char*)p - (const char*)mDeviceMemoryBase;
  return offset < mDeviceMemorySize ? (T*)((char*)mInternals->mem_gpu.contents + offset) : p;
}

template <class T, class N, class S>
inline void GPUReconstructionMetal::SortOnHost(const krnlSetupTime& _xyz, T* begin, const N& count, const S& comp)
{
  if (_xyz.z.evList) {
    StreamWaitForEvents(_xyz.x.stream, _xyz.z.evList, _xyz.z.nEvents);
  }
  DrainStream(_xyz.x.stream);
  T* hostBegin = HostPointer(begin);
  GPUCommonAlgorithm::sortDeviceDynamic(hostBegin, hostBegin + count(), comp);
  if (_xyz.z.ev) {
    RecordMarker(_xyz.z.ev, _xyz.x.stream);
  }
}

// Kernels that are a long serial chain per GPU thread, with little parallelism,
// run on the host instead, on the shared memory, as the CPU backend runs them.
// The processors in mProcessorsShadow hold GPU addresses: the kernel runs on a
// private copy of mProcessorsShadow, whose processors point at the host view of
// the same memory, with the host's parameters and calibration objects, and
// which is their constant memory. SetDevicePointers relocates the pointers in
// place, so they are copied and the shadow is restored before the kernel runs:
// the kernel, on many host threads, never touches the shadow.
struct GPUMetalHostProcessor : public GPUProcessor {
  static const GPUConstantMem*& ConstantMem(GPUProcessor& p) { return p.*&GPUMetalHostProcessor::mConstantMem; }
};

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionMetal::RunOnHost(const krnlSetupTime& _xyz, typename T::processorType* procs, uint32_t nProcs, uint32_t nBlocks, const Args&... args)
{
  if (_xyz.z.evList) {
    StreamWaitForEvents(_xyz.x.stream, _xyz.z.evList, _xyz.z.nEvents);
  }
  DrainStream(_xyz.x.stream);
  const auto start = std::chrono::steady_clock::now();
  const auto shadow = std::make_unique<char[]>(sizeof(GPUConstantMem));
  static_assert(alignof(GPUConstantMem) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
  const auto hostMem = std::make_unique<char[]>(sizeof(GPUConstantMem));
  GPUConstantMem* host = (GPUConstantMem*)hostMem.get();
  memcpy(shadow.get(), (void*)mProcessorsShadow, sizeof(GPUConstantMem));
  for (const auto& res : mMemoryResources) {
    if (res.PtrDevice()) {
      res.SetDevicePointers(HostPointer(res.PtrDevice()));
    }
  }
  memcpy((void*)host, (void*)mProcessorsShadow, sizeof(GPUConstantMem));
  for (const auto& res : mMemoryResources) {
    if (res.PtrDevice()) {
      res.SetDevicePointers(res.PtrDevice());
    }
  }
  memcpy((void*)mProcessorsShadow, shadow.get(), sizeof(GPUConstantMem));
  host->param = param();
  host->calibObjects = processors()->calibObjects;
  host->errorCodes.setMemory(HostPointer(const_cast<uint32_t*>(host->errorCodes.getErrorPtr()) - 1));
  auto* hostProcs = (typename T::processorType*)((char*)host + ((char*)procs - (char*)mProcessorsShadow));
  for (uint32_t i = 0; i < nProcs; i++) {
    GPUMetalHostProcessor::ConstantMem(hostProcs[i]) = host;
  }
  tbb::parallel_for<uint32_t>(0, nBlocks, [&](uint32_t iBlock) {
    typename T::GPUSharedMemory smem;
    T::template Thread<I>(nBlocks, 1, iBlock, 0, smem, hostProcs[_xyz.y.index], args...);
  });
  if (GetProcessingSettings().debugLevel >= 1 && memcmp(shadow.get(), (void*)mProcessorsShadow, sizeof(GPUConstantMem))) {
    GPUFatal("Metal: mProcessorsShadow changed while %s ran on the host", GetKernelName<T, I>());
  }
  _xyz.t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (_xyz.z.ev) {
    RecordMarker(_xyz.z.ev, _xyz.x.stream);
  }
}

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCGMMergerSortTracks, 0>(const krnlSetupTime& _xyz)
{
  // The fit runs one track per thread, in this order. Tracks of similar length
  // keep the threads of a SIMD group busy for as long as each other; within a
  // length bucket of 16 clusters, tracks of the same sector and side then read
  // the same pages of the cluster, transformation and calibration data, which
  // the Metal GPU, bound by its address translation in the fit, needs more than
  // the exact length order.
  const GPUTPCGMMergedTrack* trk = HostPointer(mProcessorsShadow->tpcMerger.MergedTracks());
  auto comp = [trk](const int32_t aa, const int32_t bb) {
    const GPUTPCGMMergedTrack& a = trk[aa];
    const GPUTPCGMMergedTrack& b = trk[bb];
    if (a.OK() != b.OK()) {
      return a.OK();
    }
    if (a.CCE() != b.CCE()) {
      return a.CCE() > b.CCE();
    }
    if (a.NClusters() / 16 != b.NClusters() / 16) {
      return a.NClusters() / 16 > b.NClusters() / 16;
    }
    const bool sideA = a.GetParam().GetZ() < 0, sideB = b.GetParam().GetZ() < 0;
    if (sideA != sideB || a.GetAlpha() != b.GetAlpha()) {
      return sideA != sideB ? sideA < sideB : a.GetAlpha() < b.GetAlpha();
    }
    return aa < bb;
  };
  SortOnHost(_xyz, mProcessorsShadow->tpcMerger.TrackOrderProcess(), [&] { return HostPointer(mProcessorsShadow->tpcMerger.Memory())->nMergedTracks; }, comp);
}

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCGMMergerSortTracksQPt, 0>(const krnlSetupTime& _xyz)
{
  SortOnHost(_xyz, mProcessorsShadow->tpcMerger.TrackSort(), [&] { return HostPointer(mProcessorsShadow->tpcMerger.Memory())->nMergedTracks; }, internal::GPUTPCGMMergerSortTracksQPt_comp(HostPointer(mProcessorsShadow->tpcMerger.MergedTracks())));
}

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCGMMergerMergeLoopers, 1>(const krnlSetupTime& _xyz)
{
  SortOnHost(_xyz, mProcessorsShadow->tpcMerger.LooperCandidates(), [&] { return HostPointer(mProcessorsShadow->tpcMerger.Memory())->nLooperMatchCandidates; }, internal::GPUTPCGMMergerMergeLoopers_comp());
}

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCGMO2Output, GPUTPCGMO2Output::sort>(const krnlSetupTime& _xyz)
{
  SortOnHost(_xyz, mProcessorsShadow->tpcMerger.TrackSortO2(), [&] { return HostPointer(mProcessorsShadow->tpcMerger.Memory())->nO2Tracks; }, internal::GPUTPCGMO2OutputSort_comp());
}

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCGMMergerMergeBorders, 3>(const krnlSetupTime& _xyz, gputpcgmmergertypes::GPUTPCGMBorderRange* const& range, int32_t const& N, int32_t const& cmpMax)
{
  if (cmpMax) {
    SortOnHost(_xyz, range, [&] { return N; }, internal::MergeBorderTracks_compMax());
  } else {
    SortOnHost(_xyz, range, [&] { return N; }, internal::MergeBorderTracks_compMin());
  }
}

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCGMMergerMergeBorders, 2>(const krnlSetupTime& _xyz, int32_t const& iSector, uint8_t const& mergeMode)
{
  RunOnHost<GPUTPCGMMergerMergeBorders, 2>(_xyz, &mProcessorsShadow->tpcMerger, 1, 1, iSector, mergeMode);
}

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCExtrapolationTracking, 0>(const krnlSetupTime& _xyz)
{
  RunOnHost<GPUTPCExtrapolationTracking, 0>(_xyz, mProcessorsShadow->tpcTrackers, GPUTPCGeometry::NSECTORS, getNKernelHostThreads(false));
}
