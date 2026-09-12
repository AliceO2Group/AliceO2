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

#ifndef ALICEO2_ITSMFT_TRACKING_ROFVIEWS_H_
#define ALICEO2_ITSMFT_TRACKING_ROFVIEWS_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#ifndef GPUCA_GPUCODE
#include <format>
#endif

#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "DataFormatsITS/Vertex.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

#ifndef GPUCA_GPUCODE
#include "Framework/Logger.h"
#endif

namespace o2::itsmft::tracking
{

/// Runtime timing data used by the non-owning ROF views. The detector-side
/// fixed-capacity table builders use this same type, so the timing arithmetic
/// has one implementation at the application/core boundary.
struct ROFTimingLayer {
  using BCType = o2::its::TimeStampType;
  using BCRange = o2::dataformats::RangeReference<BCType, BCType>;

  BCType mNROFsTF{0};
  BCType mROFLength{0};
  BCType mROFDelay{0};
  BCType mROFBias{0};
  BCType mROFAddTimeErr{0};

  GPUhdi() BCType getROFStartInBC(BCType rofId) const noexcept
  {
    assert(rofId < mNROFsTF && rofId >= 0);
    return (mROFLength * rofId) + mROFDelay + mROFBias;
  }

  GPUhdi() BCType getROFEndInBC(BCType rofId) const noexcept
  {
    assert(rofId < mNROFsTF);
    return getROFStartInBC(rofId) + mROFLength;
  }

  GPUhdi() o2::its::TimeEstBC getROFTimeBounds(BCType rofId, bool withError = false) const noexcept
  {
    if (withError) {
      int64_t start = getROFStartInBC(rofId);
      int64_t end = getROFEndInBC(rofId);
      start = o2::gpu::CAMath::Max(start - mROFAddTimeErr, int64_t(0));
      end += mROFAddTimeErr;
      return {static_cast<BCType>(start), static_cast<o2::its::TimeStampErrorType>(end - start)};
    }
    return {getROFStartInBC(rofId), static_cast<o2::its::TimeStampErrorType>(mROFLength)};
  }

  GPUhdi() BCType getROF(BCType bc) const noexcept
  {
    const BCType offset = mROFDelay + mROFBias;
    if (bc <= offset) {
      return 0;
    }
    return (bc - offset) / mROFLength;
  }

  GPUhdi() BCType getROF(o2::its::TimeStamp ts) const noexcept
  {
    const BCType offset = mROFDelay + mROFBias;
    const BCType bc = (ts.getTimeStamp() < ts.getTimeStampError()) ? BCType(0) : static_cast<BCType>(o2::gpu::CAMath::Floor(ts.getTimeStamp() - ts.getTimeStampError()));
    if (bc <= offset) {
      return 0;
    }
    return (bc - offset) / mROFLength;
  }

  GPUhdi() BCType getROF(float time) const noexcept
  {
    const float offset = static_cast<float>(mROFDelay + mROFBias);
    if (time <= offset) {
      return 0;
    }
    return static_cast<BCType>((time - offset) / mROFLength);
  }

  GPUhdi() bool intersectROF(BCType rof, float lower, float upper) const noexcept
  {
    const auto rofTS = getROFTimeBounds(rof, true);
    return static_cast<float>(rofTS.upper()) > lower && upper > static_cast<float>(rofTS.lower());
  }

  GPUhdi() BCRange getROFRange(o2::its::TimeStamp ts) const noexcept
  {
    return getROFRange(ts.getTimeStamp() - ts.getTimeStampError(), ts.getTimeStamp() + ts.getTimeStampError());
  }

  GPUhdi() BCRange getROFRange(o2::its::TimeEstBC ts) const noexcept
  {
    return getROFRange(static_cast<float>(ts.lower()), static_cast<float>(ts.upper()));
  }

  GPUhdi() BCRange getROFRange(float lower, float upper) const noexcept
  {
    const BCType maxROF = mNROFsTF - 1;
    BCType first = o2::gpu::CAMath::Clamp(getROF(lower - mROFAddTimeErr), BCType{0}, maxROF);
    BCType last = o2::gpu::CAMath::Clamp(getROF(upper + mROFAddTimeErr), BCType{0}, maxROF);

    if (first <= last && !intersectROF(first, lower, upper)) {
      ++first;
    }
    if (last >= first && !intersectROF(last, lower, upper)) {
      --last;
    }
    return {first, first <= last ? static_cast<BCType>(last - first + 1) : BCType{0}};
  }

#ifndef GPUCA_GPUCODE
  GPUh() std::string asString() const
  {
    return std::format("NROFsPerTF {:4} ROFLength {:4} ({:4} per Orbit) ROFDelay {:4} ROFBias {:4} ROFAddTimeErr {:4}", mNROFsTF, mROFLength, (o2::constants::lhc::LHCMaxBunches / mROFLength), mROFDelay, mROFBias, mROFAddTimeErr);
  }

  GPUh() void print() const
  {
    LOG(info) << asString();
  }
#endif
};

template <typename TableEntry, typename TableIndex>
struct ROFOverlapView {
  const TableEntry* mFlatTable{nullptr};
  const TableIndex* mIndices{nullptr};
  const ROFTimingLayer* mLayers{nullptr};
  int32_t mLayerCount{0};

  GPUhdi() const ROFTimingLayer& getLayer(int32_t layer) const noexcept
  {
    assert(layer >= 0 && layer < mLayerCount);
    return mLayers[layer];
  }

  GPUh() int32_t getClock() const noexcept
  {
    int32_t fastest = 0;
    uint32_t maxNROFs{0};
    for (int32_t iL{0}; iL < mLayerCount; ++iL) {
      const auto& layer = getLayer(iL);
      if (layer.mNROFsTF > maxNROFs) {
        fastest = iL;
        maxNROFs = layer.mNROFsTF;
      }
    }
    return fastest;
  }

  GPUh() const ROFTimingLayer& getClockLayer() const noexcept { return mLayers[getClock()]; }

  GPUhdi() const TableEntry& getOverlap(int32_t from, int32_t to, size_t rofIdx) const noexcept
  {
    assert(from < mLayerCount && to < mLayerCount);
    const auto& idx = mIndices[(from * mLayerCount) + to];
    assert(rofIdx < idx.getEntries());
    return mFlatTable[idx.getFirstEntry() + rofIdx];
  }

  GPUhdi() bool doROFsOverlap(int32_t layer0, size_t rof0, int32_t layer1, size_t rof1) const noexcept
  {
    if (layer0 == layer1) {
      return rof0 == rof1;
    }
    assert(layer0 < mLayerCount && layer1 < mLayerCount);
    const auto& idx = mIndices[(layer0 * mLayerCount) + layer1];
    if (rof0 >= idx.getEntries()) {
      return false;
    }
    const auto& overlap = mFlatTable[idx.getFirstEntry() + rof0];
    if (overlap.getEntries() == 0) {
      return false;
    }
    const size_t firstCompatible = overlap.getFirstEntry();
    const size_t lastCompatible = firstCompatible + overlap.getEntries() - 1;
    return rof1 >= firstCompatible && rof1 <= lastCompatible;
  }

  GPUhdi() o2::its::TimeEstBC getTimeStamp(int32_t layer0, size_t rof0, int32_t layer1, size_t rof1) const noexcept
  {
    assert(layer0 < mLayerCount && layer1 < mLayerCount);
    assert(doROFsOverlap(layer0, rof0, layer1, rof1));
    return mLayers[layer0].getROFTimeBounds(rof0, true) + mLayers[layer1].getROFTimeBounds(rof1, true);
  }

#ifndef GPUCA_GPUCODE
  GPUh() void printAll() const
  {
    for (int32_t i = 0; i < mLayerCount; ++i) {
      for (int32_t j = 0; j < mLayerCount; ++j) {
        if (i != j) {
          printMapping(i, j);
        }
      }
    }
    printSummary();
  }

  GPUh() void printMapping(int32_t from, int32_t to) const
  {
    if (from == to) {
      LOGP(error, "No self-lookup supported");
      return;
    }
    const auto& idx = mIndices[(from * mLayerCount) + to];
    LOGF(info, "Overlap mapping: Layer %d -> Layer %d", from, to);
    LOGP(info, "From: {}", mLayers[from].asString());
    LOGP(info, "To  : {}", mLayers[to].asString());
    for (int32_t i = 0; i < idx.getEntries(); ++i) {
      const auto& overlap = getOverlap(from, to, i);
      LOGF(info, "%d -> first %d count %d", i, overlap.getFirstEntry(), overlap.getEntries());
    }
  }

  GPUh() void printSummary() const
  {
    uint32_t totalEntries{0};
    size_t flatTableSize{0};
    for (int32_t i = 0; i < mLayerCount; ++i) {
      for (int32_t j = 0; j < mLayerCount; ++j) {
        if (i != j) {
          const auto& idx = mIndices[(i * mLayerCount) + j];
          totalEntries += idx.getEntries();
          flatTableSize += idx.getEntries();
        }
      }
    }
    LOGF(info, "Total overlap table size: %u entries", totalEntries);
    LOGF(info, "Flat table size: %zu entries", flatTableSize);
  }
#endif
};

template <typename TableEntry, typename TableIndex>
struct ROFVertexLookupView {
  const TableEntry* mFlatTable{nullptr};
  const TableIndex* mIndices{nullptr};
  const ROFTimingLayer* mLayers{nullptr};
  int32_t mLayerCount{0};

  GPUhdi() const ROFTimingLayer& getLayer(int32_t layer) const noexcept
  {
    assert(layer >= 0 && layer < mLayerCount);
    return mLayers[layer];
  }

  GPUhdi() const TableEntry& getVertices(int32_t layer, size_t rofIdx) const noexcept
  {
    assert(layer >= 0 && layer < mLayerCount);
    const auto& idx = mIndices[layer];
    assert(rofIdx < idx.getEntries());
    return mFlatTable[idx.getFirstEntry() + rofIdx];
  }

  GPUh() int32_t getMaxVerticesPerROF() const noexcept
  {
    int32_t maxCount = 0;
    for (int32_t layer = 0; layer < mLayerCount; ++layer) {
      const auto& idx = mIndices[layer];
      for (int32_t i = 0; i < idx.getEntries(); ++i) {
        maxCount = o2::gpu::CAMath::Max(maxCount, static_cast<int32_t>(mFlatTable[idx.getFirstEntry() + i].getEntries()));
      }
    }
    return maxCount;
  }

  GPUhdi() bool isVertexCompatible(int32_t layer, size_t rofIdx, const o2::its::Vertex& vertex) const noexcept
  {
    assert(layer >= 0 && layer < mLayerCount);
    const auto& layerDef = mLayers[layer];
    int64_t rofLower = o2::gpu::CAMath::Max(static_cast<int64_t>(layerDef.getROFStartInBC(rofIdx)) - static_cast<int64_t>(layerDef.mROFAddTimeErr), int64_t(0));
    int64_t rofUpper = static_cast<int64_t>(layerDef.getROFEndInBC(rofIdx)) + layerDef.mROFAddTimeErr;
    auto vLower = static_cast<int64_t>(vertex.getTimeStamp().lower());
    auto vUpper = static_cast<int64_t>(vertex.getTimeStamp().upper());
    return vUpper >= rofLower && vLower < rofUpper;
  }

#ifndef GPUCA_GPUCODE
  GPUh() void printAll() const
  {
    for (int32_t layer = 0; layer < mLayerCount; ++layer) {
      const auto& idx = mIndices[layer];
      LOGF(info, "Vertex lookup: Layer %d, ROFs %u", layer, idx.getEntries());
    }
  }
#endif
};

template <typename TableEntry, typename TableIndex>
struct ROFMaskView {
  const TableEntry* mFlatMask{nullptr};
  const TableIndex* mLayerROFOffsets{nullptr};
  int32_t mLayerCount{0};

  GPUhdi() bool isROFEnabled(int32_t layer, int32_t rofId) const noexcept
  {
    assert(layer >= 0 && layer < mLayerCount);
    return mFlatMask[mLayerROFOffsets[layer] + rofId] != 0u;
  }

#ifndef GPUCA_GPUCODE
  GPUh() void printLayer(int32_t layer) const
  {
    constexpr int wROF = 10;
    constexpr int wActive = 10;
    const int32_t nROFs = mLayerROFOffsets[layer + 1] - mLayerROFOffsets[layer];
    LOGF(info, "Mask table: Layer %d", layer);
    LOGF(info, "%*s | %*s", wROF, "ROF", wActive, "Enabled");
    LOGF(info, "%.*s-+-%.*s", wROF, "----------", wActive, "----------");
    for (int32_t rof = 0; rof < nROFs; ++rof) {
      LOGF(info, "%*d | %*d", wROF, rof, wActive, static_cast<int>(isROFEnabled(layer, rof)));
    }
  }

  GPUh() std::string asString(int32_t layer) const
  {
    const int32_t nROFs = mLayerROFOffsets[layer + 1] - mLayerROFOffsets[layer];
    int32_t enabledROFs = 0;
    for (int32_t rof = 0; rof < nROFs; ++rof) {
      if (isROFEnabled(layer, rof)) {
        ++enabledROFs;
      }
    }
    return std::format("ROFMask on Layer {} ROFs enabled: {}/{}", layer, enabledROFs, nROFs);
  }

  GPUh() void print(int32_t layer) const
  {
    LOG(info) << asString(layer);
  }

  GPUh() void printAll() const
  {
    for (int32_t layer = 0; layer < mLayerCount; ++layer) {
      printLayer(layer);
    }
  }
#endif
};

using RuntimeROFTableEntry = o2::dataformats::RangeReference<ROFTimingLayer::BCType, ROFTimingLayer::BCType>;
using RuntimeROFOverlapView = ROFOverlapView<RuntimeROFTableEntry, RuntimeROFTableEntry>;
using RuntimeROFVertexLookupView = ROFVertexLookupView<RuntimeROFTableEntry, RuntimeROFTableEntry>;
using RuntimeROFMaskView = ROFMaskView<uint8_t, uint32_t>;

/// A non-owning event view assembled by an ITS/MFT adapter. The core sees one
/// runtime context, while detector-specific fixed-capacity tables stay at the
/// adapter edge that owns their lifetime.
struct RuntimeROFViews {
  RuntimeROFOverlapView overlap{};
  RuntimeROFVertexLookupView vertexLookup{};
  RuntimeROFMaskView mask{};
  RuntimeROFMaskView upcMask{};
};

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_ROFVIEWS_H_
