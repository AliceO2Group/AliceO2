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

#include "ITSMFTTracking/detail/TimeFrameScratch.h"

#include <stdexcept>

namespace o2::itsmft::tracking
{

namespace
{
template <typename Operation, typename... Containers>
void applyToContainers(Operation&& operation, Containers&... containers)
{
  (operation(containers), ...);
}
} // namespace

void TimeFrameScratch::clearResizeEdgeStorage(std::size_t nEdges)
{
  auto clearResize = [this, nEdges](auto& container) {
    clearResizeBoundedVector(container, nEdges, mMemoryPool.get());
  };
  applyToContainers(clearResize, mTracklets, mTrackletsLookupTable, mTrackletLabels,
                    mEdgePhiCuts, mEdgeMSAngles);
}

void TimeFrameScratch::clearResizeCellStorage(std::size_t nCells)
{
  auto clearResize = [this, nCells](auto& container) {
    clearResizeBoundedVector(container, nCells, mMemoryPool.get());
  };
  applyToContainers(clearResize, mCells, mCellsLookupTable, mCellsNeighbours,
                    mCellsNeighboursTopology, mCellsNeighboursLUT, mCellLabels);
}

void TimeFrameScratch::configureStorage(std::size_t nEdges, std::size_t nCells)
{
  mNEdges = nEdges;
  mNCells = nCells;
  clearResizeEdgeStorage(nEdges);
  clearResizeCellStorage(nCells);
}

void TimeFrameScratch::reset()
{
  applyToContainers([](auto& container) { deepVectorClear(container); },
                    mTracklets, mTrackletsLookupTable, mTrackletLabels, mCells,
                    mCellsLookupTable, mCellsNeighbours, mCellsNeighboursTopology,
                    mCellsNeighboursLUT, mCellLabels, mEdgePhiCuts, mEdgeMSAngles);
}

void TimeFrameScratch::clearStorage() noexcept
{
  applyToContainers([](auto& container) { container.clear(); },
                    mTracklets, mTrackletsLookupTable, mTrackletLabels, mCells,
                    mCellsLookupTable, mCellsNeighbours, mCellsNeighboursTopology,
                    mCellsNeighboursLUT, mCellLabels);
  deepVectorClear(mEdgePhiCuts);
  deepVectorClear(mEdgeMSAngles);
  mNEdges = 0;
  mNCells = 0;
}

void TimeFrameScratch::setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool)
{
  mMemoryPool = std::move(pool);
  applyToContainers([this](auto& container) { deepVectorClear(container, mMemoryPool.get()); },
                    mEdgePhiCuts, mEdgeMSAngles, mTracklets, mTrackletsLookupTable,
                    mTrackletLabels, mCells, mCellsLookupTable, mCellsNeighbours,
                    mCellsNeighboursTopology, mCellsNeighboursLUT, mCellLabels);
}

std::size_t TimeFrameScratch::getNumberOfCells() const
{
  std::size_t result = 0;
  for (const auto& cells : mCells) {
    result += cells.size();
  }
  return result;
}

std::size_t TimeFrameScratch::getNumberOfTracklets() const
{
  std::size_t result = 0;
  for (const auto& tracklets : mTracklets) {
    result += tracklets.size();
  }
  return result;
}

std::size_t TimeFrameScratch::getNumberOfNeighbours() const
{
  std::size_t result = 0;
  for (const auto& neighbours : mCellsNeighbours) {
    result += neighbours.size();
  }
  return result;
}

void TimeFrameScratch::beginIteration(std::size_t nEdges, std::size_t nCells,
                                      gsl::span<const std::size_t> trackletLookupSizes)
{
  if (nEdges > mNEdges || nCells > mNCells || trackletLookupSizes.size() != nEdges) {
    throw std::logic_error{"TimeFrameScratch::beginIteration(): requested storage exceeds configured capacity"};
  }

  clearResizeCellStorage(nCells);
  clearResizeEdgeStorage(nEdges);

  for (std::size_t edge = 0; edge < nEdges; ++edge) {
    mTrackletsLookupTable[edge].resize(trackletLookupSizes[edge] + 1, 0);
  }
}

} // namespace o2::itsmft::tracking
