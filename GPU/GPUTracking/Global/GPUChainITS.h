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
struct TrackingFrameInfo;
class TrackITSExt;
} // namespace o2::its
namespace o2::itsmft::tracking
{
class ExternalAllocator;
} // namespace o2::itsmft::tracking

namespace o2::gpu
{
class GPUChainITS final : public GPUChain
{
  friend class GPUReconstruction;

 public:
  ~GPUChainITS() final;
  int32_t Init() override;
  int32_t PrepareEvent() override;
  int32_t Finalize() override;
  int32_t RunChain() override;

  void RegisterPermanentMemoryAndProcessors() final {};
  void RegisterGPUProcessors() final {};
  void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) final;

  o2::its::TrackerTraits<7>* GetITSTrackerTraits();
  o2::its::VertexerTraits<7>* GetITSVertexerTraits();
  o2::its::TimeFrame<7>* GetITSTimeframe();
  o2::itsmft::tracking::ExternalAllocator* GetITSMFTFrameworkAllocator();

 protected:
  GPUChainITS(GPUReconstruction* rec);
  std::unique_ptr<o2::itsmft::tracking::ExternalAllocator> mFrameworkAllocator;
  std::unique_ptr<o2::its::TimeFrame<7>> mITSTimeFrame;
  std::unique_ptr<o2::its::TrackerTraits<7>> mITSTrackerTraits;
  std::unique_ptr<o2::its::VertexerTraits<7>> mITSVertexerTraits;
};
} // namespace o2::gpu

#endif
