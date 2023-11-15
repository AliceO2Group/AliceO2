// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainITS.h
/// \author David Rohr

#ifndef GPUCHAINITS_H
#define GPUCHAINITS_H

#include "GPUChain.h"
namespace o2::its
{
struct Cluster;
template <unsigned char N>
class Road;
class Cell;
struct TrackingFrameInfo;
class TrackITSExt;
class GPUFrameworkExternalAllocator;
} // namespace o2::its

namespace GPUCA_NAMESPACE::gpu
{
class GPUChainITS : public GPUChain
{
  friend class GPUReconstruction;

 public:
  ~GPUChainITS() override;
  void RegisterPermanentMemoryAndProcessors() override;
  void RegisterGPUProcessors() override;
  int Init() override;
  int PrepareEvent() override;
  int Finalize() override;
  int RunChain() override;
  void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) override;

  o2::its::TrackerTraits* GetITSTrackerTraits();
  o2::its::VertexerTraits* GetITSVertexerTraits();
  o2::its::TimeFrame* GetITSTimeframe();

 protected:
  GPUChainITS(GPUReconstruction* rec, unsigned int maxTracks = GPUCA_MAX_ITS_FIT_TRACKS);
  std::unique_ptr<o2::its::TrackerTraits> mITSTrackerTraits;
  std::unique_ptr<o2::its::VertexerTraits> mITSVertexerTraits;
  std::unique_ptr<o2::its::TimeFrame> mITSTimeFrame;
  std::unique_ptr<o2::its::GPUFrameworkExternalAllocator> mFrameworkAllocator;

  unsigned int mMaxTracks;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
