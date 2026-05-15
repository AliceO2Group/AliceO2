// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TRACKINGITSU_INCLUDE_TRACKINGTOPOLOGY_H_
#define TRACKINGITSU_INCLUDE_TRACKINGTOPOLOGY_H_

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifndef GPUCA_GPUCODE
#include <fmt/format.h>
#include <string>
#include "Framework/Logger.h"
#endif

#include "CommonDataFormat/RangeReference.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "ITStracking/LayerMask.h"

namespace o2::its
{

template <int NLayers>
class TrackingTopology
{
 public:
  using Id = uint8_t;
  using Mask = LayerMask;
  using Range = o2::dataformats::RangeReference<Id, Id>;
  static constexpr int MaxTransitions = (NLayers * (NLayers - 1)) / 2;
  static constexpr int MaxCells = (NLayers * (NLayers - 1) * (NLayers - 2)) / 6;
  static_assert(NLayers < std::numeric_limits<Id>::max());
  static_assert(MaxTransitions <= std::numeric_limits<Id>::max());
  static_assert(MaxCells <= std::numeric_limits<Id>::max());

  // Describes from which layer to which layer the look-up happens
  struct LayerTransition {
    Id fromLayer{0};
    Id toLayer{0};
  };
  static_assert(std::is_standard_layout_v<LayerTransition>);
  static_assert(std::is_trivially_copyable_v<LayerTransition>);
  static_assert(sizeof(LayerTransition) == (2 * sizeof(Id)));

  // Describes from which LayerTransition a tracklet is allowed to originate
  // and with which LayerTransition this can be combined additionally the hitMasked is cached
  struct CellTopology {
    Id firstTransition{0};
    Id secondTransition{0};
    Mask hitLayerMask{0};
  };
  static_assert(std::is_standard_layout_v<CellTopology>);
  static_assert(std::is_trivially_copyable_v<CellTopology>);
  static_assert(sizeof(CellTopology) == (2 * sizeof(Id)) + sizeof(Mask));

  // GPU ready view of the underlying LUTs
  struct View {
    const LayerTransition* transitions{nullptr};
    const CellTopology* cells{nullptr};
    const Range* cellsByFirstTransitionIndex{nullptr};
    const Id* cellsByFirstTransition{nullptr};
    Id nTransitions{0};
    Id nCells{0};
    Id nCellsByFirstTransition{0};

    GPUhdi() const LayerTransition& getTransition(Id id) const { return transitions[id]; }
    GPUhdi() const CellTopology& getCell(Id id) const { return cells[id]; }
    GPUhdi() Range getCellsStartingWithTransition(Id transitionId) const { return cellsByFirstTransitionIndex[transitionId]; }

#ifndef GPUCA_GPUCODE
    std::string asString() const
    {
      std::string out = fmt::format("TrackingTopology: transitions={} cells={}", nTransitions, nCells);
      out += "\n  transitions:";
      for (Id transitionId = 0; transitionId < nTransitions; ++transitionId) {
        const auto& t = transitions[transitionId];
        out += fmt::format("\n    {}: {} -> {}", transitionId, t.fromLayer, t.toLayer);
      }
      out += "\n  cells:";
      for (Id cellId = 0; cellId < nCells; ++cellId) {
        const auto& c = cells[cellId];
        const auto& first = transitions[c.firstTransition];
        const auto& second = transitions[c.secondTransition];
        out += fmt::format("\n    {}: {} -> {} -> {} hitMask={} transitions=({}, {})", cellId, first.fromLayer, first.toLayer, second.toLayer, c.hitLayerMask.asString(), c.firstTransition, c.secondTransition);
      }
      return out;
    }

    void print() const
    {
      LOGP(info, "{}", asString());
    }
#endif
  };

  void init(int maxLayers, int maxHoles, Mask holeLayerMask)
  {
    clear();
    mMaxLayers = o2::gpu::CAMath::Max(0, o2::gpu::CAMath::Min(maxLayers, NLayers));
    mMaxHoles = o2::gpu::CAMath::Max(maxHoles, 0);
    mHoleLayerMask = holeLayerMask;
    for (int fromLayer = 0; fromLayer < mMaxLayers; ++fromLayer) {
      for (int toLayer = fromLayer + 1; toLayer < mMaxLayers; ++toLayer) {
        if (Mask::skipped(fromLayer, toLayer).isAllowedHoleMask(mMaxHoles, mHoleLayerMask)) {
          mTransitions[mNTransitions++] = LayerTransition{static_cast<Id>(fromLayer), static_cast<Id>(toLayer)};
        }
      }
    }

    for (Id firstId = 0; firstId < mNTransitions; ++firstId) {
      const auto& first = mTransitions[firstId];
      for (Id secondId = 0; secondId < mNTransitions; ++secondId) {
        const auto& second = mTransitions[secondId];
        if (first.toLayer != second.fromLayer) {
          continue;
        }
        const Mask hitMask{first.fromLayer, first.toLayer, second.toLayer};
        if (hitMask.isAllowed(mMaxHoles, mHoleLayerMask)) {
          mCells[mNCells++] = CellTopology{firstId, secondId, hitMask};
        }
      }
    }

    fillCellsByTransition();
  }

  View getView() const
  {
    return View{mTransitions.data(),
                mCells.data(),
                mCellsByFirstTransitionIndex.data(),
                mCellsByFirstTransition.data(),
                mNTransitions,
                mNCells,
                mNCellsByFirstTransition};
  }

  View getDeviceView(const LayerTransition* deviceTransitions,
                     const CellTopology* deviceCells,
                     const Range* deviceCellsByFirstTransitionIndex,
                     const Id* deviceCellsByFirstTransition) const
  {
    return View{deviceTransitions,
                deviceCells,
                deviceCellsByFirstTransitionIndex,
                deviceCellsByFirstTransition,
                mNTransitions,
                mNCells,
                mNCellsByFirstTransition};
  }

  const auto& getTransitions() const noexcept { return mTransitions; }
  const auto& getCells() const noexcept { return mCells; }
  const auto& getCellsByFirstTransitionIndex() const noexcept { return mCellsByFirstTransitionIndex; }
  const auto& getCellsByFirstTransition() const noexcept { return mCellsByFirstTransition; }
  Id getNTransitions() const noexcept { return mNTransitions; }
  Id getNCells() const noexcept { return mNCells; }
  Id getNCellsByFirstTransition() const noexcept { return mNCellsByFirstTransition; }

 private:
  void clear()
  {
    mNTransitions = 0;
    mNCells = 0;
    mNCellsByFirstTransition = 0;
    mTransitions.fill({});
    mCells.fill({});
    mCellsByFirstTransitionIndex.fill(Range{0, 0});
    mCellsByFirstTransition.fill(0);
  }

  void fillCellsByTransition()
  {
    std::array<Id, MaxTransitions> counts{};
    for (Id cellId = 0; cellId < mNCells; ++cellId) {
      ++counts[mCells[cellId].firstTransition];
    }

    Id offset = 0;
    for (Id transitionId = 0; transitionId < mNTransitions; ++transitionId) {
      mCellsByFirstTransitionIndex[transitionId].setFirstEntry(offset);
      mCellsByFirstTransitionIndex[transitionId].setEntries(counts[transitionId]);
      offset += counts[transitionId];
    }

    std::array<Id, MaxTransitions> cursor{};
    for (Id cellId = 0; cellId < mNCells; ++cellId) {
      const Id transitionId = mCells[cellId].firstTransition;
      mCellsByFirstTransition[mCellsByFirstTransitionIndex[transitionId].getFirstEntry() + cursor[transitionId]++] = cellId;
    }
    mNCellsByFirstTransition = offset;
  }

  int mMaxLayers{0};
  int mMaxHoles{0};
  Mask mHoleLayerMask{0};
  Id mNTransitions{0};
  Id mNCells{0};
  Id mNCellsByFirstTransition{0};
  std::array<LayerTransition, MaxTransitions> mTransitions{};
  std::array<CellTopology, MaxCells> mCells{};
  std::array<Range, MaxTransitions> mCellsByFirstTransitionIndex{};
  std::array<Id, MaxCells> mCellsByFirstTransition{};
};

} // namespace o2::its

#endif
