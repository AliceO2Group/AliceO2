// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainITS.h
/// \author David Rohr

#ifndef GPUCHAINITS_H
#define GPUCHAINITS_H

#include "GPUChain.h"
namespace o2
{
namespace its
{
class Cluster;
class Road;
class Cell;
class TrackingFrameInfo;
class TrackITS;
} // namespace its
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
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
  int RunStandalone() override;
  void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) override;

  int PrepareAndRunITSTrackFit(std::vector<o2::its::Road>& roads, std::array<const o2::its::Cluster*, 7> clusters, std::array<const o2::its::Cell*, 5> cells, const std::array<std::vector<o2::its::TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITS>& tracks);
  int RunITSTrackFit(std::vector<o2::its::Road>& roads, std::array<const o2::its::Cluster*, 7> clusters, std::array<const o2::its::Cell*, 5> cells, const std::array<std::vector<o2::its::TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITS>& tracks);

  o2::its::TrackerTraits* GetITSTrackerTraits() { return mITSTrackerTraits.get(); }
  o2::its::VertexerTraits* GetITSVertexerTraits() { return mITSVertexerTraits.get(); }

 protected:
  GPUChainITS(GPUReconstruction* rec);
  std::unique_ptr<o2::its::TrackerTraits> mITSTrackerTraits;
  std::unique_ptr<o2::its::VertexerTraits> mITSVertexerTraits;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
