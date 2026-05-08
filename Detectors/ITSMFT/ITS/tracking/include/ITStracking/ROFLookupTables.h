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

#ifndef TRACKINGITSU_INCLUDE_ROFOVERLAPTABLE_H_
#define TRACKINGITSU_INCLUDE_ROFOVERLAPTABLE_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <ranges>

#ifndef GPUCA_GPUCODE
#include <format>
#include "Framework/Logger.h"
#endif

#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "DataFormatsITS/Vertex.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2::its
{

// Layer timing definition
struct LayerTiming {
  using BCType = TimeStampType;
  BCType mNROFsTF{0};       // number of ROFs per timeframe
  BCType mROFLength{0};     // ROF length in BC
  BCType mROFDelay{0};      // delay of ROFs wrt start of first orbit in TF in BC
  BCType mROFBias{0};       // bias wrt to the LHC clock in BC
  BCType mROFAddTimeErr{0}; // additionally imposed uncertainty on ROF time in BC

  // return start of ROF in BC
  // this does not account for the opt. error!
  GPUhdi() BCType getROFStartInBC(BCType rofId) const noexcept
  {
    assert(rofId < mNROFsTF && rofId >= 0);
    return (mROFLength * rofId) + mROFDelay + mROFBias;
  }

  // return end of ROF in BCs
  // this does not account for the opt. error!
  GPUhdi() BCType getROFEndInBC(BCType rofId) const noexcept
  {
    assert(rofId < mNROFsTF);
    return getROFStartInBC(rofId) + mROFLength;
  }

  // return (clamped) time-interval of rof
  GPUhdi() TimeEstBC getROFTimeBounds(BCType rofId, bool withError = false) const noexcept
  {
    if (withError) {
      int64_t start = getROFStartInBC(rofId);
      int64_t end = getROFEndInBC(rofId);
      start = o2::gpu::CAMath::Max(start - mROFAddTimeErr, int64_t(0));
      end += mROFAddTimeErr;
      return {static_cast<BCType>(start), static_cast<TimeStampErrorType>(end - start)};
    }
    return {getROFStartInBC(rofId), static_cast<TimeStampErrorType>(mROFLength)};
  }

  // return which ROF this BC belongs to
  GPUhi() BCType getROF(BCType bc) const noexcept
  {
    const BCType offset = mROFDelay + mROFBias;
    if (bc <= offset) {
      return 0;
    }
    return (bc - offset) / mROFLength;
  }

  // return which ROF this timestamp belongs by its lower edge
  GPUhi() BCType getROF(TimeStamp ts) const noexcept
  {
    const BCType offset = mROFDelay + mROFBias;
    const BCType bc = (ts.getTimeStamp() < ts.getTimeStampError()) ? BCType(0) : static_cast<BCType>(o2::gpu::CAMath::Floor(ts.getTimeStamp() - ts.getTimeStampError()));
    if (bc <= offset) {
      return 0;
    }
    return (bc - offset) / mROFLength;
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

// Base class for lookup to define layers
template <int32_t NLayers>
class LayerTimingBase
{
 protected:
  LayerTiming mLayers[NLayers];

 public:
  using T = LayerTiming::BCType;
  LayerTimingBase() = default;

  GPUh() void defineLayer(int32_t layer, T nROFsTF, T rofLength, T rofDelay, T rofBias, T rofTE)
  {
    assert(layer >= 0 && layer < NLayers);
    mLayers[layer] = {nROFsTF, rofLength, rofDelay, rofBias, rofTE};
  }

  GPUh() void defineLayer(int32_t layer, const LayerTiming& timing)
  {
    assert(layer >= 0 && layer < NLayers);
    mLayers[layer] = timing;
  }

  GPUhdi() const LayerTiming& getLayer(int32_t layer) const
  {
    assert(layer >= 0 && layer < NLayers);
    return mLayers[layer];
  }

  GPUhdi() constexpr int32_t getEntries() noexcept { return NLayers; }

#ifndef GPUCA_GPUCODE
  GPUh() void print() const
  {
    LOGP(info, "Imposed time structure:");
    for (int32_t iL{0}; iL < NLayers; ++iL) {
      LOGP(info, "\tLayer:{} {}", iL, mLayers[iL].asString());
    }
  }
#endif
};

// GPU friendly view of the table below
template <int32_t NLayers, typename TableEntry, typename TableIndex>
struct ROFOverlapTableView {
  const TableEntry* mFlatTable{nullptr};
  const TableIndex* mIndices{nullptr};
  const LayerTiming* mLayers{nullptr};

  GPUhdi() const LayerTiming& getLayer(int32_t layer) const noexcept
  {
    assert(layer >= 0 && layer < NLayers);
    return mLayers[layer];
  }

  GPUh() int32_t getClock() const noexcept
  {
    // we take the fastest layer as clock
    int32_t fastest = 0;
    uint32_t maxNROFs{0};
    for (int32_t iL{0}; iL < NLayers; ++iL) {
      const auto& layer = getLayer(iL);
      // by definition the fastest layer has the most ROFs
      // this also solves the problem of a delay large than ROFLength
      // if mNROFsTF is correct
      if (layer.mNROFsTF > maxNROFs) {
        fastest = iL;
        maxNROFs = layer.mNROFsTF;
      }
    }
    return fastest;
  }

  GPUh() const LayerTiming& getClockLayer() const noexcept
  {
    return mLayers[getClock()];
  }

  GPUhdi() const TableEntry& getOverlap(int32_t from, int32_t to, size_t rofIdx) const noexcept
  {
    assert(from < NLayers && to < NLayers);
    const size_t linearIdx = (from * NLayers) + to;
    const auto& idx = mIndices[linearIdx];
    assert(rofIdx < idx.getEntries());
    return mFlatTable[idx.getFirstEntry() + rofIdx];
  }

  GPUhdi() bool doROFsOverlap(int32_t layer0, size_t rof0, int32_t layer1, size_t rof1) const noexcept
  {
    if (layer0 == layer1) { // layer is compatible with itself
      return rof0 == rof1;
    }

    assert(layer0 < NLayers && layer1 < NLayers);
    const size_t linearIdx = (layer0 * NLayers) + layer1;
    const auto& idx = mIndices[linearIdx];

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

  GPUhdi() TimeEstBC getTimeStamp(int32_t layer0, size_t rof0, int32_t layer1, size_t rof1) const noexcept
  {
    assert(layer0 < NLayers && layer1 < NLayers);
    assert(doROFsOverlap(layer0, rof0, layer1, rof1));
    // retrieves the combined timestamp
    // e.g., taking one cluster from rof0 and one from rof1
    //       and constructing a tracklet (doublet) what is its time
    // this assumes that the rofs overlap, e.g. doROFsOverlap -> true
    // get timestamp including margins from rof0 and rof1
    const auto t0 = mLayers[layer0].getROFTimeBounds(rof0, true);
    const auto t1 = mLayers[layer1].getROFTimeBounds(rof1, true);
    return t0 + t1;
  }

#ifndef GPUCA_GPUCODE
  /// Print functions
  GPUh() void printAll() const
  {
    for (int32_t i = 0; i < NLayers; ++i) {
      for (int32_t j = 0; j < NLayers; ++j) {
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

    constexpr int w_index = 10;
    constexpr int w_first = 12;
    constexpr int w_last = 12;
    constexpr int w_count = 10;

    LOGF(info, "Overlap mapping: Layer %d -> Layer %d", from, to);
    LOGP(info, "From: {}", mLayers[from].asString());
    LOGP(info, "To  : {}", mLayers[to].asString());
    LOGF(info, "%*s | %*s | %*s | %*s", w_index, "ROF.index", w_first, "First.ROF", w_last, "Last.ROF", w_count, "Count");
    LOGF(info, "%.*s-+-%.*s-+-%.*s-+-%.*s", w_index, "----------", w_first, "------------", w_last, "------------", w_count, "----------");

    const size_t linearIdx = (from * NLayers) + to;
    const auto& idx = mIndices[linearIdx];
    for (int32_t i = 0; i < idx.getEntries(); ++i) {
      const auto& overlap = getOverlap(from, to, i);
      LOGF(info, "%*d | %*d | %*d | %*d", w_index, i, w_first, overlap.getFirstEntry(), w_last, overlap.getEntriesBound() - 1, w_count, overlap.getEntries());
    }
  }

  GPUh() void printSummary() const
  {
    uint32_t totalEntries{0};
    size_t flatTableSize{0};

    for (int32_t i = 0; i < NLayers; ++i) {
      for (int32_t j = 0; j < NLayers; ++j) {
        if (i != j) {
          const size_t linearIdx = (i * NLayers) + j;
          const auto& idx = mIndices[linearIdx];
          totalEntries += idx.getEntries();
          flatTableSize += idx.getEntries();
        }
      }
    }

    for (int32_t i = 0; i < NLayers; ++i) {
      mLayers[i].print();
    }

    const uint32_t totalBytes = (flatTableSize * sizeof(TableEntry)) + (static_cast<unsigned long>(NLayers * NLayers) * sizeof(TableIndex));
    LOGF(info, "------------------------------------------------------------");
    LOGF(info, "Total overlap table size: %u entries", totalEntries);
    LOGF(info, "Flat table size: %zu entries", flatTableSize);
    LOGF(info, "Total view size: %u bytes", totalBytes);
    LOGF(info, "------------------------------------------------------------");
  }
#endif
};

// Precalculated lookup table to find overlapping ROFs in another layer given a ROF index in the current layer
template <int32_t NLayers>
class ROFOverlapTable : public LayerTimingBase<NLayers>
{
 public:
  using T = LayerTimingBase<NLayers>::T;
  using TableEntry = dataformats::RangeReference<T, T>;
  using TableIndex = dataformats::RangeReference<T, T>;

  using View = ROFOverlapTableView<NLayers, TableEntry, TableIndex>;
  ROFOverlapTable() = default;

  GPUh() void init()
  {
    std::vector<TableEntry> table[NLayers][NLayers];
    for (int32_t i{0}; i < NLayers; ++i) {
      for (int32_t j{0}; j < NLayers; ++j) {
        if (i != j) { // we do not need self-lookup
          buildMapping(i, j, table[i][j]);
        }
      }
    }
    flatten(table);
  }

  GPUh() View getView() const
  {
    View view;
    view.mFlatTable = mFlatTable.data();
    view.mIndices = mIndices;
    view.mLayers = this->mLayers;
    return view;
  }

  GPUh() View getDeviceView(const TableEntry* deviceFlatTablePtr, const TableIndex* deviceIndicesPtr, const LayerTiming* deviceLayerTimingPtr) const
  {
    View view;
    view.mFlatTable = deviceFlatTablePtr;
    view.mIndices = deviceIndicesPtr;
    view.mLayers = deviceLayerTimingPtr;
    return view;
  }

  GPUh() size_t getFlatTableSize() const noexcept { return mFlatTable.size(); }
  static GPUh() constexpr size_t getIndicesSize() { return static_cast<size_t>(NLayers * NLayers); }

 private:
  GPUh() void buildMapping(int32_t from, int32_t to, std::vector<TableEntry>& table)
  {
    const auto& layerFrom = this->mLayers[from];
    const auto& layerTo = this->mLayers[to];
    table.resize(layerFrom.mNROFsTF);

    for (int32_t iROF{0}; iROF < layerFrom.mNROFsTF; ++iROF) {
      int64_t fromStart = o2::gpu::CAMath::Max((int64_t)layerFrom.getROFStartInBC(iROF) - (int64_t)layerFrom.mROFAddTimeErr, int64_t(0));
      int64_t fromEnd = (int64_t)layerFrom.getROFEndInBC(iROF) + layerFrom.mROFAddTimeErr;

      int32_t firstROFTo = o2::gpu::CAMath::Max(0, (int32_t)((fromStart - (int64_t)layerTo.mROFAddTimeErr - (int64_t)layerTo.mROFDelay - (int64_t)layerTo.mROFBias) / (int64_t)layerTo.mROFLength));
      auto lastROFTo = (int32_t)((fromEnd + (int64_t)layerTo.mROFAddTimeErr - (int64_t)layerTo.mROFDelay - (int64_t)layerTo.mROFBias - 1) / (int64_t)layerTo.mROFLength);
      firstROFTo = o2::gpu::CAMath::Max(0, firstROFTo);
      lastROFTo = o2::gpu::CAMath::Min((int32_t)layerTo.mNROFsTF - 1, lastROFTo);

      while (firstROFTo <= lastROFTo) {
        int64_t toStart = o2::gpu::CAMath::Max((int64_t)layerTo.getROFStartInBC(firstROFTo) - (int64_t)layerTo.mROFAddTimeErr, int64_t(0));
        int64_t toEnd = (int64_t)layerTo.getROFEndInBC(firstROFTo) + layerTo.mROFAddTimeErr;
        if (toEnd > fromStart && toStart < fromEnd) {
          break;
        }
        ++firstROFTo;
      }
      while (lastROFTo >= firstROFTo) {
        int64_t toStart = o2::gpu::CAMath::Max((int64_t)layerTo.getROFStartInBC(lastROFTo) - (int64_t)layerTo.mROFAddTimeErr, int64_t(0));
        int64_t toEnd = (int64_t)layerTo.getROFEndInBC(lastROFTo) + layerTo.mROFAddTimeErr;
        if (toEnd > fromStart && toStart < fromEnd) {
          break;
        }
        --lastROFTo;
      }
      int32_t count = (firstROFTo <= lastROFTo) ? (lastROFTo - firstROFTo + 1) : 0;
      table[iROF] = {static_cast<T>(firstROFTo), static_cast<T>(count)};
    }
  }

  GPUh() void flatten(const std::vector<TableEntry> table[NLayers][NLayers])
  {
    size_t total{0};
    for (int32_t i{0}; i < NLayers; ++i) {
      for (int32_t j{0}; j < NLayers; ++j) {
        if (i != j) { // we do not need self-lookup
          total += table[i][j].size();
        }
      }
    }

    mFlatTable.reserve(total);

    for (int32_t i{0}; i < NLayers; ++i) {
      for (int32_t j{0}; j < NLayers; ++j) {
        size_t idx = (i * NLayers) + j;
        if (i != j) {
          mIndices[idx].setFirstEntry(static_cast<T>(mFlatTable.size()));
          mIndices[idx].setEntries(static_cast<T>(table[i][j].size()));
          mFlatTable.insert(mFlatTable.end(), table[i][j].begin(), table[i][j].end());
        } else {
          mIndices[idx] = {0, 0};
        }
      }
    }
  }

  TableIndex mIndices[NLayers * NLayers];
  std::vector<TableEntry> mFlatTable;
};

// GPU friendly view of the table below
template <int32_t NLayers, typename TableEntry, typename TableIndex>
struct ROFVertexLookupTableView {
  const TableEntry* mFlatTable{nullptr};
  const TableIndex* mIndices{nullptr};
  const LayerTiming* mLayers{nullptr};

  GPUhdi() const LayerTiming& getLayer(int32_t layer) const noexcept
  {
    assert(layer >= 0 && layer < NLayers);
    return mLayers[layer];
  }

  GPUhdi() const TableEntry& getVertices(int32_t layer, size_t rofIdx) const noexcept
  {
    assert(layer < NLayers);
    const auto& idx = mIndices[layer];
    assert(rofIdx < idx.getEntries());
    return mFlatTable[idx.getFirstEntry() + rofIdx];
  }

  GPUh() int32_t getMaxVerticesPerROF() const noexcept
  {
    int32_t maxCount = 0;
    for (int32_t layer = 0; layer < NLayers; ++layer) {
      const auto& idx = mIndices[layer];
      for (int32_t i = 0; i < idx.getEntries(); ++i) {
        const auto& entry = mFlatTable[idx.getFirstEntry() + i];
        maxCount = o2::gpu::CAMath::Max(maxCount, static_cast<int32_t>(entry.getEntries()));
      }
    }
    return maxCount;
  }

  // Check if a specific vertex is compatible with a given ROF
  GPUhdi() bool isVertexCompatible(int32_t layer, size_t rofIdx, const Vertex& vertex) const noexcept
  {
    assert(layer < NLayers);
    const auto& layerDef = mLayers[layer];
    int64_t rofLower = o2::gpu::CAMath::Max((int64_t)layerDef.getROFStartInBC(rofIdx) - (int64_t)layerDef.mROFAddTimeErr, int64_t(0));
    int64_t rofUpper = (int64_t)layerDef.getROFEndInBC(rofIdx) + layerDef.mROFAddTimeErr;
    auto vLower = (int64_t)vertex.getTimeStamp().lower();
    auto vUpper = (int64_t)vertex.getTimeStamp().upper();
    return vUpper >= rofLower && vLower < rofUpper;
  }

#ifndef GPUCA_GPUCODE
  GPUh() void printAll() const
  {
    for (int32_t i = 0; i < NLayers; ++i) {
      printLayer(i);
    }
    printSummary();
  }

  GPUh() void printLayer(int32_t layer) const
  {
    constexpr int w_rof = 10;
    constexpr int w_first = 12;
    constexpr int w_last = 12;
    constexpr int w_count = 10;

    LOGF(info, "Vertex lookup: Layer %d", layer);
    LOGF(info, "%*s | %*s | %*s | %*s", w_rof, "ROF.index", w_first, "First.Vtx", w_last, "Last.Vtx", w_count, "Count");
    LOGF(info, "%.*s-+-%.*s-+-%.*s-+-%.*s", w_rof, "----------", w_first, "------------", w_last, "------------", w_count, "----------");

    const auto& idx = mIndices[layer];
    for (int32_t i = 0; i < idx.getEntries(); ++i) {
      const auto& entry = mFlatTable[idx.getFirstEntry() + i];
      int first = entry.getFirstEntry();
      int count = entry.getEntries();
      int last = first + count - 1;
      LOGF(info, "%*d | %*d | %*d | %*d", w_rof, i, w_first, first, w_last, last, w_count, count);
    }
  }

  GPUh() void printSummary() const
  {
    uint32_t totalROFs{0};
    uint32_t totalVertexRefs{0};

    for (int32_t i = 0; i < NLayers; ++i) {
      const auto& idx = mIndices[i];
      totalROFs += idx.getEntries();

      for (int32_t j = 0; j < idx.getEntries(); ++j) {
        const auto& entry = mFlatTable[idx.getFirstEntry() + j];
        totalVertexRefs += entry.getEntries();
      }
    }

    const uint32_t totalBytes = (totalROFs * sizeof(TableEntry)) + (NLayers * sizeof(TableIndex));
    LOGF(info, "------------------------------------------------------------");
    LOGF(info, "Total ROFs in table: %u", totalROFs);
    LOGF(info, "Total vertex references: %u", totalVertexRefs);
    LOGF(info, "Total view size: %u bytes", totalBytes);
    LOGF(info, "------------------------------------------------------------");
  }
#endif
};

// Precalculated lookup table to find vertices compatible with ROFs
// Given a layer and ROF index, returns the range of vertices that overlap in time.
// The vertex time is defined as symmetrical [t0-e,t0+e]
// It needs to be guaranteed that the input vertices are sorted by their lower-bound!
// additionally compatibliyty has to be queried per vertex!
template <int32_t NLayers>
class ROFVertexLookupTable : public LayerTimingBase<NLayers>
{
 public:
  using T = LayerTimingBase<NLayers>::T;
  using BCType = LayerTiming::BCType;
  using TableEntry = dataformats::RangeReference<T, T>;
  using TableIndex = dataformats::RangeReference<T, T>;
  using View = ROFVertexLookupTableView<NLayers, TableEntry, TableIndex>;

  ROFVertexLookupTable() = default;

  GPUh() size_t getFlatTableSize() const noexcept { return mFlatTable.size(); }
  static GPUh() constexpr size_t getIndicesSize() { return NLayers; }

  // Build the lookup table given a sorted array of vertices
  // vertices must be sorted by timestamp, then by error (secondary)
  GPUh() void init(const Vertex* vertices, size_t nVertices)
  {
    if (nVertices > std::numeric_limits<T>::max()) {
      LOGF(fatal, "too many vertices %zu, max supported is %u", nVertices, std::numeric_limits<T>::max());
    }

    std::vector<TableEntry> table[NLayers];
    for (int32_t layer{0}; layer < NLayers; ++layer) {
      buildMapping(layer, vertices, nVertices, table[layer]);
    }
    flatten(table);
  }

  // Pre-allocated needed memory, then use update(...)
  GPUh() void init()
  {
    size_t total{0};
    for (int32_t layer{0}; layer < NLayers; ++layer) {
      total += this->mLayers[layer].mNROFsTF;
    }
    mFlatTable.resize(total, {0, 0});
    size_t offset = 0;
    for (int32_t layer{0}; layer < NLayers; ++layer) {
      size_t nROFs = this->mLayers[layer].mNROFsTF;
      mIndices[layer].setFirstEntry(static_cast<T>(offset));
      mIndices[layer].setEntries(static_cast<T>(nROFs));
      offset += nROFs;
    }
  }

  // Recalculate lookup table with new vertices
  GPUh() void update(const Vertex* vertices, size_t nVertices)
  {
    size_t offset = 0;
    for (int32_t layer{0}; layer < NLayers; ++layer) {
      const auto& idx = mIndices[layer];
      size_t nROFs = idx.getEntries();
      for (size_t iROF = 0; iROF < nROFs; ++iROF) {
        updateROFMapping(layer, iROF, vertices, nVertices, offset + iROF);
      }
      offset += nROFs;
    }
  }

  GPUh() View getView() const
  {
    View view;
    view.mFlatTable = mFlatTable.data();
    view.mIndices = mIndices;
    view.mLayers = this->mLayers;
    return view;
  }

  GPUh() View getDeviceView(const TableEntry* deviceFlatTablePtr, const TableIndex* deviceIndicesPtr, const LayerTiming* deviceLayerTimingPtr) const
  {
    View view;
    view.mFlatTable = deviceFlatTablePtr;
    view.mIndices = deviceIndicesPtr;
    view.mLayers = deviceLayerTimingPtr;
    return view;
  }

 private:
  // Build the mapping for one layer
  GPUh() void buildMapping(int32_t layer, const Vertex* vertices, size_t nVertices, std::vector<TableEntry>& table)
  {
    const auto& layerDef = this->mLayers[layer];
    table.resize(layerDef.mNROFsTF);
    size_t vertexSearchStart = 0;
    for (int32_t iROF{0}; iROF < layerDef.mNROFsTF; ++iROF) {
      int64_t rofLower = o2::gpu::CAMath::Max((int64_t)layerDef.getROFStartInBC(iROF) - (int64_t)layerDef.mROFAddTimeErr, int64_t(0));
      int64_t rofUpper = (int64_t)layerDef.getROFEndInBC(iROF) + layerDef.mROFAddTimeErr;
      size_t lastVertex = binarySearchFirst(vertices, nVertices, vertexSearchStart, rofUpper);
      size_t firstVertex = vertexSearchStart;
      while (firstVertex < lastVertex) {
        auto vUpper = (int64_t)vertices[firstVertex].getTimeStamp().upper();
        if (vUpper > rofLower) {
          break;
        }
        ++firstVertex;
      }
      size_t count = (lastVertex > firstVertex) ? (lastVertex - firstVertex) : 0;
      table[iROF] = {static_cast<T>(firstVertex), static_cast<T>(count)};
      vertexSearchStart = firstVertex;
    }
  }

  // Update a single ROF's vertex mapping
  GPUh() void updateROFMapping(int32_t layer, size_t iROF, const Vertex* vertices, size_t nVertices, size_t flatTableIdx)
  {
    const auto& layerDef = this->mLayers[layer];
    int64_t rofLower = o2::gpu::CAMath::Max((int64_t)layerDef.getROFStartInBC(iROF) - (int64_t)layerDef.mROFAddTimeErr, int64_t(0));
    int64_t rofUpper = (int64_t)layerDef.getROFEndInBC(iROF) + layerDef.mROFAddTimeErr;
    size_t lastVertex = binarySearchFirst(vertices, nVertices, 0, rofUpper);
    size_t firstVertex = 0;
    while (firstVertex < lastVertex) {
      int64_t vUpper = (int64_t)vertices[firstVertex].getTimeStamp().getTimeStamp() +
                       (int64_t)vertices[firstVertex].getTimeStamp().getTimeStampError();
      if (vUpper > rofLower) {
        break;
      }
      ++firstVertex;
    }
    size_t count = (lastVertex > firstVertex) ? (lastVertex - firstVertex) : 0;
    mFlatTable[flatTableIdx].setFirstEntry(static_cast<T>(firstVertex));
    mFlatTable[flatTableIdx].setEntries(static_cast<T>(count));
  }

  // Binary search for first vertex where maxBC >= targetBC
  GPUh() size_t binarySearchFirst(const Vertex* vertices, size_t nVertices, size_t searchStart, BCType targetBC) const
  {
    size_t left = searchStart;
    size_t right = nVertices;
    while (left < right) {
      size_t mid = left + ((right - left) / 2);
      int64_t lower = (int64_t)vertices[mid].getTimeStamp().getTimeStamp() -
                      (int64_t)vertices[mid].getTimeStamp().getTimeStampError();
      if (lower < targetBC) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  }

  // Compress the temporary table into a single flat table
  GPUh() void flatten(const std::vector<TableEntry> table[NLayers])
  {
    // Count total entries
    size_t total{0};
    for (int32_t i{0}; i < NLayers; ++i) {
      total += table[i].size();
    }

    mFlatTable.reserve(total);

    // Build flat table and indices
    for (int32_t i{0}; i < NLayers; ++i) {
      mIndices[i].setFirstEntry(static_cast<T>(mFlatTable.size()));
      mIndices[i].setEntries(static_cast<T>(table[i].size()));
      mFlatTable.insert(mFlatTable.end(), table[i].begin(), table[i].end());
    }
  }

  TableIndex mIndices[NLayers];
  std::vector<TableEntry> mFlatTable;
};

// GPU-friendly view of the ROF mask table
template <int32_t NLayers, typename TableEntry, typename TableIndex>
struct ROFMaskTableView {
  const TableEntry* mFlatMask{nullptr};
  const TableIndex* mLayerROFOffsets{nullptr}; // size NLayers+1

  GPUhdi() bool isROFEnabled(int32_t layer, int32_t rofId) const noexcept
  {
    assert(layer >= 0 && layer < NLayers);
    return mFlatMask[mLayerROFOffsets[layer] + rofId] != 0u;
  }

#ifndef GPUCA_GPUCODE
  GPUh() void printAll() const
  {
    for (int32_t i = 0; i < NLayers; ++i) {
      printLayer(i);
    }
  }

  GPUh() void printLayer(int32_t layer) const
  {
    constexpr int w_rof = 10;
    constexpr int w_active = 10;
    int32_t nROFs = mLayerROFOffsets[layer + 1] - mLayerROFOffsets[layer];
    LOGF(info, "Mask table: Layer %d", layer);
    LOGF(info, "%*s | %*s", w_rof, "ROF", w_active, "Enabled");
    LOGF(info, "%.*s-+-%.*s", w_rof, "----------", w_active, "----------");
    for (int32_t i = 0; i < nROFs; ++i) {
      LOGF(info, "%*d | %*d", w_rof, i, w_active, (int)isROFEnabled(layer, i));
    }
  }

  GPUh() std::string asString(int32_t layer) const
  {
    int32_t nROFs = mLayerROFOffsets[layer + 1] - mLayerROFOffsets[layer];
    int32_t enabledROFs = 0;
    for (int32_t j = 0; j < nROFs; ++j) {
      if (isROFEnabled(layer, j)) {
        ++enabledROFs;
      }
    }
    return std::format("ROFMask on Layer {} ROFs enabled: {}/{}", layer, enabledROFs, nROFs);
  }

  GPUh() void print(int32_t layer) const
  {
    LOG(info) << asString(layer);
  }
#endif
};

// Per-ROF per-layer boolean mask (uint8_t for GPU compatibility).
template <int32_t NLayers>
class ROFMaskTable : public LayerTimingBase<NLayers>
{
 public:
  using T = LayerTimingBase<NLayers>::T;
  using BCRange = dataformats::RangeReference<T, T>;
  using TableIndex = uint32_t;
  using TableEntry = uint8_t;
  using View = ROFMaskTableView<NLayers, TableEntry, TableIndex>;

  ROFMaskTable() = default;
  GPUh() explicit ROFMaskTable(const LayerTimingBase<NLayers>& timingBase) : LayerTimingBase<NLayers>(timingBase) { init(); }

  GPUh() void init()
  {
    int32_t totalROFs = 0;
    for (int32_t layer{0}; layer < NLayers; ++layer) {
      mLayerROFOffsets[layer] = totalROFs;
      totalROFs += this->getLayer(layer).mNROFsTF;
    }
    mLayerROFOffsets[NLayers] = totalROFs; // sentinel
    mFlatMask.resize(totalROFs, 0u);
  }

  GPUh() size_t getFlatMaskSize() const noexcept { return mFlatMask.size(); }

  GPUh() void setROFEnabled(int32_t layer, int32_t rofId, uint8_t state = 1) noexcept
  {
    assert(layer >= 0 && layer < NLayers);
    assert(rofId >= 0 && rofId < mLayerROFOffsets[layer + 1] - mLayerROFOffsets[layer]);
    mFlatMask[mLayerROFOffsets[layer] + rofId] = state;
  }

  GPUh() void setROFsEnabled(int32_t layer, int32_t firstRof, int32_t nRofs, uint8_t state = 1) noexcept
  {
    assert(layer >= 0 && layer < NLayers);
    assert(firstRof >= 0);
    assert(firstRof + nRofs <= mLayerROFOffsets[layer + 1] - mLayerROFOffsets[layer]);
    std::memset(mFlatMask.data() + mLayerROFOffsets[layer] + firstRof, state, nRofs);
  }

  // Enable all ROFs in all layers that are time-compatible with the given BC range
  GPUh() void selectROF(const BCRange& t)
  {
    const int32_t bcStart = t.getFirstEntry();
    const int32_t bcEnd = t.getEntriesBound();
    for (int32_t layer{0}; layer < NLayers; ++layer) {
      const auto& lay = this->getLayer(layer);
      const int32_t offset = mLayerROFOffsets[layer];
      for (int32_t rofId{0}; rofId < lay.mNROFsTF; ++rofId) {
        if (static_cast<int32_t>(lay.getROFStartInBC(rofId)) < bcEnd &&
            static_cast<int32_t>(lay.getROFEndInBC(rofId)) > bcStart) {
          mFlatMask[offset + rofId] = 1u;
        }
      }
    }
  }

  // Reset mask to 0, then enable all ROFs compatible with any of the given BC ranges
  GPUh() void selectROFs(const std::vector<BCRange>& ts)
  {
    resetMask();
    for (const auto& t : ts) {
      selectROF(t);
    }
  }

  GPUh() void resetMask(uint8_t s = 0u)
  {
    std::memset(mFlatMask.data(), s, mFlatMask.size());
  }

  GPUh() void invertMask()
  {
    std::ranges::transform(mFlatMask, mFlatMask.begin(), [](uint8_t x) { return 1 - x; });
  }

  GPUh() void swap(ROFMaskTable& other) noexcept
  {
    std::swap(mFlatMask, other.mFlatMask);
    std::swap(mLayerROFOffsets, other.mLayerROFOffsets);
  }

  GPUh() View getView() const
  {
    View view;
    view.mFlatMask = mFlatMask.data();
    view.mLayerROFOffsets = mLayerROFOffsets;
    return view;
  }

  GPUh() View getDeviceView(const TableEntry* deviceFlatMaskPtr, const TableIndex* deviceOffsetPtr) const
  {
    View view;
    view.mFlatMask = deviceFlatMaskPtr;
    view.mLayerROFOffsets = deviceOffsetPtr;
    return view;
  }

 private:
  TableIndex mLayerROFOffsets[NLayers + 1] = {0};
  std::vector<TableEntry> mFlatMask;
};

} // namespace o2::its

#endif
