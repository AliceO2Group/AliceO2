// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainEC0.h
/// \author David Rohr

#ifndef GPUCHAINEC0_H
#define GPUCHAINEC0_H

#include "GPUChain.h"
#include "EC0tracking/TrackerTraits.h"

namespace o2::its
{
class TrackITSExt;
} // namespace o2::its

namespace o2::ecl
{
class Cluster;
class Road;
class Cell;
class TrackingFrameInfo;
} // namespace o2::ecl


namespace GPUCA_NAMESPACE::gpu
{
class GPUChainEC0 : public GPUChain
{
  friend class GPUReconstruction;
  //using TrackITSExt = o2::its::TrackITSExt;
  //using TrackerTraits = o2::ecl::TrackerTraits;
  //using VertexerTraits = o2::ecl::VertexerTraits;

 public:
  ~GPUChainEC0() override;
  void RegisterPermanentMemoryAndProcessors() override;
  void RegisterGPUProcessors() override;
  int Init() override;
  int PrepareEvent() override;
  int Finalize() override;
  int RunChain() override;
  void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) override;

  int PrepareAndRunEC0TrackFit(std::vector<o2::ecl::Road>& roads, std::array<const o2::ecl::Cluster*, 7> clusters, std::array<const o2::ecl::Cell*, 5> cells, const std::array<std::vector<o2::ecl::TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITSExt>& tracks);
  int RunEC0TrackFit(std::vector<o2::ecl::Road>& roads, std::array<const o2::ecl::Cluster*, 7> clusters, std::array<const o2::ecl::Cell*, 5> cells, const std::array<std::vector<o2::ecl::TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITSExt>& tracks);

  o2::ecl::TrackerTraits* GetEC0TrackerTraits();
  o2::ecl::VertexerTraits* GetEC0VertexerTraits();

 protected:
  GPUChainEC0(GPUReconstruction* rec, unsigned int maxTracks = GPUCA_MAX_ITS_FIT_TRACKS);
  std::unique_ptr<o2::ecl::TrackerTraits> mEC0TrackerTraits;
  std::unique_ptr<o2::ecl::VertexerTraits> mEC0VertexerTraits;

  unsigned int mMaxTracks;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
