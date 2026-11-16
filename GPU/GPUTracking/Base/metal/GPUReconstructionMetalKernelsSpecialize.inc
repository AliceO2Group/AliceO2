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

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUTPCGMMergerSortTracks, 0>(const krnlSetupTime& _xyz)
{
  SortOnHost(_xyz, mProcessorsShadow->tpcMerger.TrackOrderProcess(), [&] { return HostPointer(mProcessorsShadow->tpcMerger.Memory())->nMergedTracks; }, internal::GPUTPCGMMergerSortTracks_comp(HostPointer(mProcessorsShadow->tpcMerger.MergedTracks())));
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
