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

/// \file GPUTPCGMMergerRebuildInternal.h
/// \author David Rohr

#ifndef GPUTPCGMMERGERREBUILDINTERNAL_H
#define GPUTPCGMMERGERREBUILDINTERNAL_H

#include "GPUTPCGMMerger.h"
#include "GPUCommonMath.h"
#include "GPUErrors.h"

namespace o2::gpu
{

struct GPUTPCGMMerger::trackRebuildHelperCandidate { // Per track, row and candidate, the row is implicit
  uint16_t idHigh, idLow;                            // 2 x 16 bit instead of uint32_t for a size of 6 bytes without packing, high part first. Combined value: 0: empty (in the first candidate: row not searched), 1: row searched but no candidate, otherwise (sector << 24) | (id + 2), with id the index of the cluster in its sector row (below 2^24 - 2)
  CAMath::half_t error;
  GPUdi() uint32_t getValue() const { return ((uint32_t)idHigh << 16) | idLow; }
  GPUdi() void setValue(uint32_t v)
  {
    idHigh = v >> 16;
    idLow = v & 0xFFFF;
  }
  GPUdi() bool isEmpty() const { return getValue() == 0; }
  GPUdi() bool hasCluster() const { return getValue() >= 2; }
  GPUdi() void setNoCandidate() { setValue(1); }
  GPUdi() uint32_t getId() const { return (getValue() & 0xFFFFFF) - 2; } // Only with hasCluster(), convert with GPUTPCGMMerger::CandidateId / CandidateClusterId
  GPUdi() uint8_t getSector() const { return idHigh >> 8; }              // Only with hasCluster()
  GPUdi() void set(uint32_t id, uint8_t sector, CAMath::half_t e)
  {
    setValue(((uint32_t)sector << 24) | (id + 2));
    error = e;
  }
};
static_assert(sizeof(GPUTPCGMMerger::trackRebuildHelperCandidate) == 6);

struct __attribute__((packed)) GPUTPCGMMerger::trackRebuildHelperRowInfo { // Per track, only with GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY >= 1
  uint32_t firstRowIndex;                                                  // Index of firstRow of the track in mTrackRebuildHelperRow and (times the number of candidates) in mClusterCandidates
  uint8_t firstRow, lastRow;                                               // Row range the rebuild can use for the track: 1 = estimate from PrepareForFit1, 2 = range filled by the candidate search of the fit, empty (firstRow > lastRow) otherwise
};

struct GPUTPCGMMerger::trackRebuildHelper { // Per track
  uint8_t reverse;
  uint8_t highInclRowLow;
  uint8_t highInclRowHigh;
};

struct GPUTPCGMMerger::trackRebuildHelperRow { // Per track and row: weight (lower 28 bits) and best (upper 4 bits)
  uint32_t data;
  enum : uint32_t { bestRowRemoved = 14,                  // Row removed (extrapolation reverted)
                    bestNoCandidate = 15 };               // No acceptable candidate
  GPUdi() uint32_t getBest() const { return data >> 28; } // 0: undecided, 1 - 13: index of selected candidate + 1, or bestRowRemoved / bestNoCandidate
  GPUdi() void setBest(uint32_t best) { data = (data & 0x0FFFFFFF) | (best << 28); }
  GPUdi() int32_t getWeight() const { return data & 0x0FFFFFFF; }
  GPUdi() int32_t setWeight(float w) { return (data = (data & 0xF0000000) | (w > 0.5f ? (w < 2147483648.f ? CAMath::Float2UIntReint(w) - 0x3F000000 : 0x0FFFFFFF) : 0)) & 0x0FFFFFFF; }
};

template <int32_t tmpMode>
struct GPUTPCGMMergerRebuildTmpRows { // Empty unless tmpMode == 2, used as GPUTPCGMMergerRebuildTmp
};
template <>
struct GPUTPCGMMergerRebuildTmpRows<2> { // Rows of one track during the candidate search of the fit with GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY == 2 (host only), the filled row range is stored to the merger after the fit
  GPUTPCGMMerger::trackRebuildHelperRow rows[GPUTPCGeometry::NROWS];
  GPUTPCGMMerger::trackRebuildHelperCandidate candidates[GPUTPCGeometry::NROWS * (GPUTPCGMMerger::trackRebuildHelperRow::bestRowRemoved - 1)]; // For the maximum number of candidates, only the part for rebuildTrackInFitClusterCandidates is used
};

GPUdi() uint32_t GPUTPCGMMerger::RebuildFirstRow(int32_t iTrk) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY >= 1) {
    return mTrackRebuildHelperRowInfo[iTrk].firstRow;
  } else {
    return 0;
  }
}

GPUdi() uint32_t GPUTPCGMMerger::RebuildLastRow(int32_t iTrk) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY >= 1) {
    return mTrackRebuildHelperRowInfo[iTrk].lastRow;
  } else {
    return GPUTPCGeometry::NROWS - 1;
  }
}

GPUdi() bool GPUTPCGMMerger::RebuildRowInRange(int32_t iTrk, uint32_t row) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY >= 1) {
    return row >= mTrackRebuildHelperRowInfo[iTrk].firstRow && row <= mTrackRebuildHelperRowInfo[iTrk].lastRow; // The range can be empty (firstRow > lastRow)
  } else {
    return true; // No guard needed, the memory covers all rows
  }
}

GPUdi() GPUTPCGMMerger::trackRebuildHelperCandidate* GPUTPCGMMerger::ClusterCandidates(int32_t iTrk, uint32_t row) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY >= 1) {
    return &mClusterCandidates[(mTrackRebuildHelperRowInfo[iTrk].firstRowIndex + row - mTrackRebuildHelperRowInfo[iTrk].firstRow) * Param().rec.tpc.rebuildTrackInFitClusterCandidates];
  } else {
    return &mClusterCandidates[(iTrk * GPUTPCGeometry::NROWS + row) * Param().rec.tpc.rebuildTrackInFitClusterCandidates];
  }
}

GPUdi() GPUTPCGMMerger::trackRebuildHelperRow& GPUTPCGMMerger::TrackRebuildHelperRow(int32_t iTrk, uint32_t row) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY >= 1) {
    return mTrackRebuildHelperRow[mTrackRebuildHelperRowInfo[iTrk].firstRowIndex + row - mTrackRebuildHelperRowInfo[iTrk].firstRow];
  } else {
    return mTrackRebuildHelperRow[iTrk * GPUTPCGeometry::NROWS + row];
  }
}

template <class T> // T = GPUTPCGMMergerRebuildTmp, template only to discard the code if it is empty
GPUdi() bool GPUTPCGMMerger::RebuildRowInRange(int32_t iTrk, uint32_t row, T* tmpRows) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY == 2) {
    if (tmpRows) {
      return true;
    }
  }
  return RebuildRowInRange(iTrk, row);
}

template <class T> // T = GPUTPCGMMergerRebuildTmp, template only to discard the code if it is empty
GPUdi() GPUTPCGMMerger::trackRebuildHelperCandidate* GPUTPCGMMerger::ClusterCandidates(int32_t iTrk, uint32_t row, T* tmpRows) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY == 2) {
    if (tmpRows) {
      return &tmpRows->candidates[row * Param().rec.tpc.rebuildTrackInFitClusterCandidates];
    }
  }
  return ClusterCandidates(iTrk, row);
}

template <class T> // T = GPUTPCGMMergerRebuildTmp, template only to discard the code if it is empty
GPUdi() GPUTPCGMMerger::trackRebuildHelperRow& GPUTPCGMMerger::TrackRebuildHelperRow(int32_t iTrk, uint32_t row, T* tmpRows) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY == 2) {
    if (tmpRows) {
      return tmpRows->rows[row];
    }
  }
  return TrackRebuildHelperRow(iTrk, row);
}

template <class T> // T = GPUTPCGMMergerRebuildTmp, template only to discard the code if it is empty
GPUdi() void GPUTPCGMMerger::ClearRebuildTmpRows(int32_t iTrk, T* tmpRows) const
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY == 2) {
    mTrackRebuildHelperRowInfo[iTrk] = trackRebuildHelperRowInfo{0, 1, 0}; // Empty row range, set by StoreRebuildTmpRows
    for (uint32_t i = 0; i < GPUTPCGeometry::NROWS; i++) {
      tmpRows->rows[i] = trackRebuildHelperRow{};
    }
    for (uint32_t i = 0; i < GPUTPCGeometry::NROWS * Param().rec.tpc.rebuildTrackInFitClusterCandidates; i++) {
      tmpRows->candidates[i] = trackRebuildHelperCandidate{};
    }
  }
}

template <class T>                                                               // T = GPUTPCGMMergerRebuildTmp, template only to discard the code if it is empty
GPUdi() bool GPUTPCGMMerger::StoreRebuildTmpRows(int32_t iTrk, const T* tmpRows) // Returns false if the rebuild memory is full. Stores the row range of the track: the rows filled by the fit and the rows of its clusters (PrepareHitWeights adds those as candidates)
{
  if constexpr (GPUCA_PAR_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY == 2) {
    const uint32_t nCandidates = Param().rec.tpc.rebuildTrackInFitClusterCandidates;
    const auto& GPUrestrict() trk = mMergedTracks[iTrk];
    int32_t first = GPUTPCGeometry::NROWS, last = -1;
    for (uint32_t i = 0; i < trk.NClusters(); i++) {
      first = CAMath::Min<int32_t>(first, mClusters[trk.FirstClusterRef() + i].row);
      last = CAMath::Max<int32_t>(last, mClusters[trk.FirstClusterRef() + i].row);
    }
    for (int32_t i = 0; i < first; i++) {
      if (tmpRows->rows[i].data != 0 || !tmpRows->candidates[i * nCandidates].isEmpty()) {
        first = i;
      }
    }
    for (int32_t i = GPUTPCGeometry::NROWS - 1; i > last; i--) {
      if (tmpRows->rows[i].data != 0 || !tmpRows->candidates[i * nCandidates].isEmpty()) {
        last = i;
      }
    }
    if (first > last) {
      return true; // Range stays empty
    }
    const uint32_t index = CAMath::AtomicAdd(&mMemory->nRebuildTrackRow, (uint32_t)(last - first + 1));
    if (index + last - first + 1 > mNRebuildTrackRow) {
      raiseError(GPUErrors::ERROR_MERGER_REBUILD_ROW_OVERFLOW, index + last - first + 1, mNRebuildTrackRow);
      return false; // Range stays empty
    }
    auto& GPUrestrict() info = mTrackRebuildHelperRowInfo[iTrk];
    info.firstRow = first;
    info.lastRow = last;
    info.firstRowIndex = index;
    for (int32_t i = 0; i <= last - first; i++) {
      mTrackRebuildHelperRow[info.firstRowIndex + i] = tmpRows->rows[first + i];
    }
    for (uint32_t i = 0; i < (last - first + 1) * nCandidates; i++) {
      mClusterCandidates[info.firstRowIndex * nCandidates + i] = tmpRows->candidates[first * nCandidates + i];
    }
  }
  return true;
}

} // namespace o2::gpu

#endif
