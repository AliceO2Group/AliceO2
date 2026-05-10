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

template <int NLayers>
class VertexerTraits
{
  using IndexTableUtilsN = IndexTableUtils<NLayers>;
  using TimeFrameN = TimeFrame<NLayers>;

 public:
  VertexerTraits() = default;
  virtual ~VertexerTraits() = default;

  // virtual vertexer interface
  virtual void initialise(const TrackingParameters& trackingParams);
  virtual void computeTracklets(const int iteration);
  virtual void computeTrackletMatching(const int iteration);
  virtual void computeVertices(const int iteration);
  virtual void adoptTimeFrame(TimeFrameN* tf) noexcept { mTimeFrame = tf; }
  virtual void updateVertexingParameters(const std::vector<VertexingParameters>& vrtPar);

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
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool) { mMemoryPool = pool; }

  static VertexLabel computeMain(const bounded_vector<o2::MCCompLabel>& elements)
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
    if (maxCount <= 1) { // need >50%
      elem.setFakeFlag();
    }
    return std::make_pair(elem, static_cast<float>(maxCount) / static_cast<float>(elements.size()));
  }

 protected:
  std::vector<VertexingParameters> mVrtParams;
  IndexTableUtilsN mIndexTableUtils;

  // Frame related quantities
  TimeFrameN* mTimeFrame = nullptr; // observer ptr
 private:
  bool skipROF(int iteration, int rof) const;

  std::shared_ptr<BoundedMemoryResource> mMemoryPool;
  std::shared_ptr<tbb::task_arena> mTaskArena;
};

} // namespace its
} // namespace o2
#endif
