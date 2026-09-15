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
/// \file TimeFrameScratch.h
/// \brief Runtime-plan-owned, detector-neutral CA workspace.
///
/// Host storage follows the runtime surface graph; device capacities remain
/// fixed. TimeFrame owns the workspace, while adapters own raw ROFs.
#ifndef ALICEO2_ITSMFT_TRACKING_TimeFrameScratch_H_
#define ALICEO2_ITSMFT_TRACKING_TimeFrameScratch_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "ITSMFTTracking/Cell.h"
#include "ITSMFTTracking/TrackingPrimitives.h"
#include "ITSMFTTracking/BoundedAllocator.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::itsmft::tracking
{

/// Detector-neutral CA state rebuilt for each tracking iteration. Operations
/// receive scalar sizes and spans; this type never depends on TimeFrame.
class TimeFrameScratch
{
 private:
  // Pool must outlive allocator-backed members.
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;

 public:
  TimeFrameScratch() = default;
  ~TimeFrameScratch() = default;
  TimeFrameScratch(const TimeFrameScratch&) = delete;
  TimeFrameScratch& operator=(const TimeFrameScratch&) = delete;
  TimeFrameScratch(TimeFrameScratch&&) = delete;
  TimeFrameScratch& operator=(TimeFrameScratch&&) = delete;

  /// Size reusable edge and cell storage; setMemoryPool() comes first.
  void configureStorage(std::size_t nEdges, std::size_t nCells);
  void beginIteration(std::size_t nEdges, std::size_t nCells,
                      gsl::span<const std::size_t> trackletLookupSizes);
  std::size_t getNEdges() const noexcept { return mNEdges; }
  std::size_t getNCells() const noexcept { return mNCells; }

  /// Clear iteration state without changing plan sizes.
  void reset();

  /// Release plan-sized storage while preserving this object's identity.
  void clearStorage() noexcept;

  /// Reseat allocator-backed containers.
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool);
  auto& getMemoryPool() const noexcept { return mMemoryPool; }
  float getEdgePhiCut(int edgeId) const { return mEdgePhiCuts[edgeId]; }
  float getEdgeMSAngle(int edgeId) const { return mEdgeMSAngles[edgeId]; }
  auto& getEdgePhiCuts() { return mEdgePhiCuts; }
  auto& getEdgeMSAngles() { return mEdgeMSAngles; }
  auto& getTrackletsLabel(int layer) { return mTrackletLabels[layer]; }
  auto& getCellsLabel(int layer) { return mCellLabels[layer]; }

  auto& getTracklets() { return mTracklets; }
  auto& getTrackletsLookupTable() { return mTrackletsLookupTable; }

  auto& getCells() { return mCells; }
  const auto& getCells() const { return mCells; }

  auto& getCellsLookupTable() { return mCellsLookupTable; }
  auto& getCellsNeighbours() { return mCellsNeighbours; }
  auto& getCellsNeighboursTopology() { return mCellsNeighboursTopology; }
  auto& getCellsNeighboursLUT() { return mCellsNeighboursLUT; }
  size_t getNumberOfCells() const;
  size_t getNumberOfTracklets() const;
  size_t getNumberOfNeighbours() const;

  // ---- Per-iteration surface and CA construction state ----
  std::vector<bounded_vector<Tracklet>> mTracklets;
  std::vector<bounded_vector<int>> mTrackletsLookupTable;
  std::vector<bounded_vector<o2::MCCompLabel>> mTrackletLabels;
  bounded_vector<float> mEdgePhiCuts;
  bounded_vector<float> mEdgeMSAngles;
  std::vector<bounded_vector<CellSeed>> mCells;
  std::vector<bounded_vector<int>> mCellsLookupTable;
  std::vector<bounded_vector<int>> mCellsNeighbours;
  std::vector<bounded_vector<int>> mCellsNeighboursTopology;
  std::vector<bounded_vector<int>> mCellsNeighboursLUT;
  std::vector<bounded_vector<o2::MCCompLabel>> mCellLabels;

 private:
  void clearResizeEdgeStorage(std::size_t nEdges);
  void clearResizeCellStorage(std::size_t nCells);

  std::size_t mNEdges{0};
  std::size_t mNCells{0};
};

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_TimeFrameScratch_H_ */
