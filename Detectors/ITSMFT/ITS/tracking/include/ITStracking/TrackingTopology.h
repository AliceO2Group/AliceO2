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
  static constexpr int MaxLinks = (NLayers * (NLayers - 1)) / 2;
  static constexpr int MaxCells = (NLayers * (NLayers - 1) * (NLayers - 2)) / 6;
  static_assert(NLayers < std::numeric_limits<Id>::max());
  static_assert(MaxLinks <= std::numeric_limits<Id>::max());
  static_assert(MaxCells <= std::numeric_limits<Id>::max());

  // Describes from which layer to which layer the look-up happens
  struct LayerLink {
    Id fromLayer{0};
    Id toLayer{0};
  };
  static_assert(std::is_standard_layout_v<LayerLink>);
  static_assert(std::is_trivially_copyable_v<LayerLink>);
  static_assert(sizeof(LayerLink) == (2 * sizeof(Id)));

  // Describes from which LayerLink a tracklet is allowed to originate
  // and with which LayerLink this can be combined additionally the hitMasked is cached
  struct CellTopology {
    Id firstLink{0};
    Id secondLink{0};
    Mask hitLayerMask{0};
  };
  static_assert(std::is_standard_layout_v<CellTopology>);
  static_assert(std::is_trivially_copyable_v<CellTopology>);
  static_assert(sizeof(CellTopology) == (2 * sizeof(Id)) + sizeof(Mask));

  // GPU ready view of the underlying LUTs
  struct View {
    const LayerLink* links{nullptr};
    const CellTopology* cells{nullptr};
    const Range* cellsByFirstLinkIndex{nullptr};
    const Id* cellsByFirstLink{nullptr};
    Mask seedingLayerMask{0};
    Id nLinks{0};
    Id nCells{0};
    Id nCellsByFirstLink{0};

    GPUhdi() const LayerLink& getLink(Id id) const { return links[id]; }
    GPUhdi() const CellTopology& getCell(Id id) const { return cells[id]; }
    GPUhdi() Range getCellsStartingWithLink(Id linkId) const { return cellsByFirstLinkIndex[linkId]; }

#ifndef GPUCA_GPUCODE
    std::string asString() const
    {
      std::string out = fmt::format("TrackingTopology: links={} cells={} seedingLayers={}", nLinks, nCells, seedingLayerMask.asString());
      out += "\n  links:";
      for (Id linkId = 0; linkId < nLinks; ++linkId) {
        const auto& t = links[linkId];
        out += fmt::format("\n    {}: {} -> {}", linkId, t.fromLayer, t.toLayer);
      }
      out += "\n  cells:";
      for (Id cellId = 0; cellId < nCells; ++cellId) {
        const auto& c = cells[cellId];
        const auto& first = links[c.firstLink];
        const auto& second = links[c.secondLink];
        out += fmt::format("\n    {}: {} -> {} -> {} hitMask={} links=({}, {})", cellId, first.fromLayer, first.toLayer, second.toLayer, c.hitLayerMask.asString(), c.firstLink, c.secondLink);
      }
      return out;
    }

    void print() const
    {
      LOGP(info, "{}", asString());
    }
#endif
  };

  void init(int maxLayers, int maxHoles, Mask holeLayerMask, Mask seedingLayerMask = 0)
  {
    clear();
    mMaxLayers = o2::gpu::CAMath::Max(0, o2::gpu::CAMath::Min(maxLayers, NLayers));
    mMaxHoles = o2::gpu::CAMath::Max(maxHoles, 0);
    mHoleLayerMask = holeLayerMask;
    mSeedingLayerMask = seedingLayerMask.empty() ? Mask::span(0, mMaxLayers - 1) : (seedingLayerMask & Mask::span(0, mMaxLayers - 1));
#ifndef GPUCA_GPUCODE
    if (mSeedingLayerMask.count() < constants::ClustersPerCell) {
      LOGP(fatal, "Tracking topology has {} seeding layers, but at least {} are required to build CA cells", mSeedingLayerMask.count(), constants::ClustersPerCell);
    }
#endif
    for (int fromLayer = 0; fromLayer < mMaxLayers; ++fromLayer) {
      if (!mSeedingLayerMask.has(fromLayer)) {
        continue;
      }
      for (int toLayer = fromLayer + 1; toLayer < mMaxLayers; ++toLayer) {
        if (mSeedingLayerMask.has(toLayer) && isAllowedSeedingLink(fromLayer, toLayer)) {
          mLinks[mNLinks++] = LayerLink{static_cast<Id>(fromLayer), static_cast<Id>(toLayer)};
        }
      }
    }

    for (Id firstId = 0; firstId < mNLinks; ++firstId) {
      const auto& first = mLinks[firstId];
      for (Id secondId = 0; secondId < mNLinks; ++secondId) {
        const auto& second = mLinks[secondId];
        if (first.toLayer != second.fromLayer) {
          continue;
        }
        const Mask hitMask{first.fromLayer, first.toLayer, second.toLayer};
        if ((hitMask.holeMask() & mSeedingLayerMask).isAllowedHoleMask(mMaxHoles, mHoleLayerMask)) {
          mCells[mNCells++] = CellTopology{firstId, secondId, hitMask};
        }
      }
    }

    fillCellsByLink();
  }

  View getView() const
  {
    return View{mLinks.data(),
                mCells.data(),
                mCellsByFirstLinkIndex.data(),
                mCellsByFirstLink.data(),
                mSeedingLayerMask,
                mNLinks,
                mNCells,
                mNCellsByFirstLink};
  }

  View getDeviceView(const LayerLink* deviceLinks,
                     const CellTopology* deviceCells,
                     const Range* deviceCellsByFirstLinkIndex,
                     const Id* deviceCellsByFirstLink) const
  {
    return View{deviceLinks,
                deviceCells,
                deviceCellsByFirstLinkIndex,
                deviceCellsByFirstLink,
                mSeedingLayerMask,
                mNLinks,
                mNCells,
                mNCellsByFirstLink};
  }

  const auto& getLinks() const noexcept { return mLinks; }
  const auto& getCells() const noexcept { return mCells; }
  const auto& getCellsByFirstLinkIndex() const noexcept { return mCellsByFirstLinkIndex; }
  const auto& getCellsByFirstLink() const noexcept { return mCellsByFirstLink; }
  Id getNLinks() const noexcept { return mNLinks; }
  Id getNCells() const noexcept { return mNCells; }
  Id getNCellsByFirstLink() const noexcept { return mNCellsByFirstLink; }

 private:
  void clear()
  {
    mNLinks = 0;
    mNCells = 0;
    mNCellsByFirstLink = 0;
    mLinks.fill({});
    mCells.fill({});
    mCellsByFirstLinkIndex.fill(Range{0, 0});
    mCellsByFirstLink.fill(0);
  }

  void fillCellsByLink()
  {
    std::array<Id, MaxLinks> counts{};
    for (Id cellId = 0; cellId < mNCells; ++cellId) {
      ++counts[mCells[cellId].firstLink];
    }

    Id offset = 0;
    for (Id linkId = 0; linkId < mNLinks; ++linkId) {
      mCellsByFirstLinkIndex[linkId].setFirstEntry(offset);
      mCellsByFirstLinkIndex[linkId].setEntries(counts[linkId]);
      offset += counts[linkId];
    }

    std::array<Id, MaxLinks> cursor{};
    for (Id cellId = 0; cellId < mNCells; ++cellId) {
      const Id linkId = mCells[cellId].firstLink;
      mCellsByFirstLink[mCellsByFirstLinkIndex[linkId].getFirstEntry() + cursor[linkId]++] = cellId;
    }
    mNCellsByFirstLink = offset;
  }

  bool isAllowedSeedingLink(int fromLayer, int toLayer) const noexcept
  {
    return (Mask::skipped(fromLayer, toLayer) & mSeedingLayerMask).isAllowedHoleMask(mMaxHoles, mHoleLayerMask);
  }

  int mMaxLayers{0};
  int mMaxHoles{0};
  Mask mHoleLayerMask{0};
  Mask mSeedingLayerMask{0};
  Id mNLinks{0};
  Id mNCells{0};
  Id mNCellsByFirstLink{0};
  std::array<LayerLink, MaxLinks> mLinks{};
  std::array<CellTopology, MaxCells> mCells{};
  std::array<Range, MaxLinks> mCellsByFirstLinkIndex{};
  std::array<Id, MaxCells> mCellsByFirstLink{};
};

} // namespace o2::its

#endif
