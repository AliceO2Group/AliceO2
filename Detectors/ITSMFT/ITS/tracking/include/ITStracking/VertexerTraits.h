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
///
/// \file VertexerTraits.h
/// \brief Class to compute the primary vertex in ITS from tracklets
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_VERTEXER_TRAITS_H_
#define O2_ITS_TRACKING_VERTEXER_TRAITS_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/MathUtils.h"

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

#include <oneapi/tbb/task_arena.h>

namespace o2
{
class MCCompLabel;

namespace its
{

enum class TrackletMode {
  Layer0Layer1 = 0,
  Layer1Layer2 = 2
};

template <int nLayers>
class VertexerTraits
{
  using IndexTableUtilsN = IndexTableUtils<nLayers>;
  using TimeFrameN = TimeFrame<nLayers>;

 public:
  VertexerTraits() = default;
  virtual ~VertexerTraits() = default;

  GPUhdi() static consteval int4 getEmptyBinsRect()
  {
    return int4{0, 0, 0, 0};
  }
  GPUhd() const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);
  GPUhd() static const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi, const IndexTableUtilsN&);
  GPUhd() static const int2 getPhiBins(float phi, float deltaPhi, const IndexTableUtilsN&);
  GPUhd() const int2 getPhiBins(float phi, float deltaPhi) { return getPhiBins(phi, deltaPhi, mIndexTableUtils); }

  // virtual vertexer interface
  virtual void initialise(const TrackingParameters& trackingParams, const int iteration = 0);
  virtual void computeTracklets(const int iteration = 0);
  virtual void computeTrackletMatching(const int iteration = 0);
  virtual void computeVertices(const int iteration = 0);
  virtual void adoptTimeFrame(TimeFrameN* tf) noexcept { mTimeFrame = tf; }
  virtual void updateVertexingParameters(const std::vector<VertexingParameters>& vrtPar, const TimeFrameGPUParameters& gpuTfPar);

  // truth tracking
  void addTruthSeedingVertices();

  // utils
  auto& getVertexingParameters() { return mVrtParams; }
  auto getVertexingParameters() const { return mVrtParams; }
  void setVertexingParameters(std::vector<VertexingParameters>& vertParams) { mVrtParams = vertParams; }
  void setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena);
  int getNThreads() { return mTaskArena->max_concurrency(); }
  virtual bool isGPU() const noexcept { return false; }
  virtual const char* getName() const noexcept { return "CPU"; }
  virtual bool usesMemoryPool() const noexcept { return true; }
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource>& pool) { mMemoryPool = pool; }

  static std::pair<o2::MCCompLabel, float> computeMain(const bounded_vector<o2::MCCompLabel>& elements)
  {
    // we only care about the source&event of the tracks, not the trackId
    auto composeVtxLabel = [](const o2::MCCompLabel& lbl) -> o2::MCCompLabel {
      return {o2::MCCompLabel::maxTrackID(), lbl.getEventID(), lbl.getSourceID(), lbl.isFake()};
    };
    std::unordered_map<o2::MCCompLabel, size_t> frequency;
    for (const auto& element : elements) {
      ++frequency[composeVtxLabel(element)];
    }
    o2::MCCompLabel elem{};
    size_t maxCount = 0;
    for (const auto& [key, count] : frequency) {
      if (count > maxCount) {
        maxCount = count;
        elem = key;
      }
    }
    return std::make_pair(elem, static_cast<float>(maxCount) / static_cast<float>(elements.size()));
  }

 protected:
  std::vector<VertexingParameters> mVrtParams;
  IndexTableUtilsN mIndexTableUtils;

  // Frame related quantities
  TimeFrameN* mTimeFrame = nullptr; // observer ptr
 private:
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;
  std::shared_ptr<tbb::task_arena> mTaskArena;

  // debug output
  void debugComputeTracklets(int iteration);
  void debugComputeTrackletMatching(int iteration);
  void debugComputeVertices(int iteration);
};

template <int nLayers>
inline void VertexerTraits<nLayers>::initialise(const TrackingParameters& trackingParams, const int iteration)
{
  mTimeFrame->initialise(0, trackingParams, 3, (bool)(!iteration)); // iteration for initialisation must be 0 for correctly resetting the frame, we need to pass the non-reset flag for vertices as well, tho.
}

template <int nLayers>
GPUhdi() const int2 VertexerTraits<nLayers>::getPhiBins(float phi, float dPhi, const IndexTableUtilsN& utils)
{
  return int2{utils.getPhiBinIndex(math_utils::getNormalizedPhi(phi - dPhi)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phi + dPhi))};
}

template <int nLayers>
GPUhdi() const int4 VertexerTraits<nLayers>::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                         const float directionZIntersection, float maxdeltaz, float maxdeltaphi,
                                                         const IndexTableUtilsN& utils)
{
  const float zRangeMin = directionZIntersection - 2 * maxdeltaz;
  const float phiRangeMin = currentCluster.phi - maxdeltaphi;
  const float zRangeMax = directionZIntersection + 2 * maxdeltaz;
  const float phiRangeMax = currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -utils.getLayerZ(layerIndex + 1) ||
      zRangeMin > utils.getLayerZ(layerIndex + 1) || zRangeMin > zRangeMax) {
    return getEmptyBinsRect();
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(utils.getNzBins() - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

template <int nLayers>
GPUhdi() const int4 VertexerTraits<nLayers>::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                         const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  return VertexerTraits::getBinsRect(currentCluster, layerIndex, directionZIntersection, maxdeltaz, maxdeltaphi, mIndexTableUtils);
}

} // namespace its
} // namespace o2
#endif
