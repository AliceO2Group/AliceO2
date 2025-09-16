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

/// \file GPUTPCTrackingData.h
/// \author Matthias Kretz, Sergey Gorbunov, David Rohr

#ifndef GPUTPCSECTORDATA_H
#define GPUTPCSECTORDATA_H

#include "GPUTPCDef.h"
#include "GPUTPCRow.h"
#include "GPUCommonMath.h"
#include "GPUParam.h"
#include "GPUProcessor.h"

namespace o2::gpu
{
struct GPUTPCClusterData;
class GPUTPCHit;

class GPUTPCTrackingData
{
 public:
  GPUTPCTrackingData() = default;

#ifndef GPUCA_GPUCODE_DEVICE
  ~GPUTPCTrackingData() = default;
  void InitializeRows(const GPUParam& p);
  void SetMaxData();
  void SetClusterData(int32_t nClusters, int32_t clusterIdOffset);
  void* SetPointersScratch(void* mem, bool idsOnGPU);
  void* SetPointersLinks(void* mem);
  void* SetPointersWeights(void* mem);
  void* SetPointersClusterIds(void* mem, bool idsOnGPU);
  void* SetPointersRows(void* mem);
#endif

  GPUd() int32_t InitFromClusterData(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUconstantref() const GPUConstantMem* mem, int32_t iSector, float* tmpMinMax);

  /**
   * Return the number of hits in this sector.
   */
  GPUhd() int32_t NumberOfHits() const { return mNumberOfHits; }
  GPUhd() int32_t NumberOfHitsPlusAlign() const { return mNumberOfHitsPlusAlign; }
  GPUhd() int32_t ClusterIdOffset() const { return mClusterIdOffset; }

  /**
   * Access to the hit links.
   *
   * The links values give the hit index in the row above/below. Or -1 if there is no link.
   */
  GPUd() calink HitLinkUpData(const GPUTPCRow& row, const calink& hitIndex) const;
  GPUd() calink HitLinkDownData(const GPUTPCRow& row, const calink& hitIndex) const;

  GPUhdi() GPUglobalref() const cahit2* HitData(const GPUTPCRow& row) const { return &mHitData[row.mHitNumberOffset]; }
  GPUhdi() GPUglobalref() cahit2* HitData(const GPUTPCRow& row) { return &mHitData[row.mHitNumberOffset]; }
  GPUhd() GPUglobalref() const cahit2* HitData() const { return (mHitData); }
  GPUdi() GPUglobalref() const calink* HitLinkUpData(const GPUTPCRow& row) const { return &mLinkUpData[row.mHitNumberOffset]; }
  GPUdi() GPUglobalref() calink* HitLinkUpData(const GPUTPCRow& row) { return &mLinkUpData[row.mHitNumberOffset]; }
  GPUdi() GPUglobalref() const calink* HitLinkDownData(const GPUTPCRow& row) const { return &mLinkDownData[row.mHitNumberOffset]; }
  GPUdi() GPUglobalref() const calink* FirstHitInBin(const GPUTPCRow& row) const { return &mFirstHitInBin[row.mFirstHitInBinOffset]; }

  GPUd() void SetHitLinkUpData(const GPUTPCRow& row, const calink& hitIndex, const calink& value);
  GPUd() void SetHitLinkDownData(const GPUTPCRow& row, const calink& hitIndex, const calink& value);

  /**
   * Return the y and z coordinate(s) of the given hit(s).
   */
  GPUd() cahit HitDataY(const GPUTPCRow& row, const uint32_t& hitIndex) const;
  GPUd() cahit HitDataZ(const GPUTPCRow& row, const uint32_t& hitIndex) const;
  GPUd() cahit2 HitData(const GPUTPCRow& row, const uint32_t& hitIndex) const;

  /**
   * For a given bin index, content tells how many hits there are in the preceding bins. This maps
   * directly to the hit index in the given row.
   *
   * \param binIndexes in the range 0 to row.Grid.N + row.Grid.Ny + 3.
   */
  GPUd() calink FirstHitInBin(const GPUTPCRow& row, calink binIndex) const;

  /**
   * If the given weight is higher than what is currently stored replace with the new weight.
   */
  GPUd() void MaximizeHitWeight(const GPUTPCRow& row, uint32_t hitIndex, uint32_t weight);
  GPUd() void SetHitWeight(const GPUTPCRow& row, uint32_t hitIndex, uint32_t weight);

  /**
   * Return the maximal weight the given hit got from one tracklet
   */
  GPUd() int32_t HitWeight(const GPUTPCRow& row, uint32_t hitIndex) const;

  /**
   * Returns the index in the original GPUTPCClusterData object of the given hit
   */
  GPUhd() int32_t ClusterDataIndex(const GPUTPCRow& row, uint32_t hitIndex) const;
  GPUd() GPUglobalref() const int32_t* ClusterDataIndex() const { return mClusterDataIndex; }
  GPUd() GPUglobalref() int32_t* ClusterDataIndex() { return mClusterDataIndex; }

  /**
   * Return the row object for the given row index.
   */
  GPUhdi() GPUglobalref() const GPUTPCRow& Row(int32_t rowIndex) const { return mRows[rowIndex]; }
  GPUhdi() GPUglobalref() GPUTPCRow* Rows() const { return mRows; }

  GPUhdi() GPUglobalref() GPUAtomic(uint32_t) * HitWeights() { return (mHitWeights); }

 private:
#ifndef GPUCA_GPUCODE
  GPUTPCTrackingData& operator=(const GPUTPCTrackingData&) = delete; // ROOT 5 tries to use this if it is not private
  GPUTPCTrackingData(const GPUTPCTrackingData&) = delete;            //
#endif
  GPUd() void CreateGrid(GPUconstantref() const GPUConstantMem* mem, GPUTPCRow* GPUrestrict() row, float yMin, float yMax, float zMin, float zMax);
  GPUd() void SetRowGridEmpty(GPUTPCRow& GPUrestrict() row);
  GPUd() static void GetMaxNBins(GPUconstantref() const GPUConstantMem* mem, GPUTPCRow* GPUrestrict() row, int32_t& maxY, int32_t& maxZ);
  GPUd() uint32_t GetGridSize(uint32_t nHits, uint32_t nRows);

  friend class GPUTPCNeighboursFinder;
  friend class GPUTPCStartHitsFinder;

  int32_t mNumberOfHits = 0; // the number of hits in this sector
  int32_t mNumberOfHitsPlusAlign = 0;
  int32_t mClusterIdOffset = 0;

  GPUglobalref() GPUTPCRow* mRows = nullptr; // The row objects needed for most accessor functions

  GPUglobalref() calink* mLinkUpData = nullptr;        // hit index in the row above which is linked to the given (global) hit index
  GPUglobalref() calink* mLinkDownData = nullptr;      // hit index in the row below which is linked to the given (global) hit index
  GPUglobalref() cahit2* mHitData = nullptr;           // packed y,z coordinate of the given (global) hit index
  GPUglobalref() int32_t* mClusterDataIndex = nullptr; // see ClusterDataIndex()

  /*
   * The size of the array is row.Grid.N + row.Grid.Ny + 3. The row.Grid.Ny + 3 is an optimization
   * to remove the need for bounds checking. The last values are the same as the entry at [N - 1].
   */
  GPUglobalref() calink* mFirstHitInBin;            // see FirstHitInBin
  GPUglobalref() GPUAtomic(uint32_t) * mHitWeights; // the weight of the longest tracklet crossed the cluster
};

GPUdi() calink GPUTPCTrackingData::HitLinkUpData(const GPUTPCRow& row, const calink& hitIndex) const { return mLinkUpData[row.mHitNumberOffset + hitIndex]; }

GPUdi() calink GPUTPCTrackingData::HitLinkDownData(const GPUTPCRow& row, const calink& hitIndex) const { return mLinkDownData[row.mHitNumberOffset + hitIndex]; }

GPUdi() void GPUTPCTrackingData::SetHitLinkUpData(const GPUTPCRow& row, const calink& hitIndex, const calink& value)
{
  mLinkUpData[row.mHitNumberOffset + hitIndex] = value;
}

GPUdi() void GPUTPCTrackingData::SetHitLinkDownData(const GPUTPCRow& row, const calink& hitIndex, const calink& value)
{
  mLinkDownData[row.mHitNumberOffset + hitIndex] = value;
}

GPUdi() cahit GPUTPCTrackingData::HitDataY(const GPUTPCRow& row, const uint32_t& hitIndex) const { return mHitData[row.mHitNumberOffset + hitIndex].x; }

GPUdi() cahit GPUTPCTrackingData::HitDataZ(const GPUTPCRow& row, const uint32_t& hitIndex) const { return mHitData[row.mHitNumberOffset + hitIndex].y; }

GPUdi() cahit2 GPUTPCTrackingData::HitData(const GPUTPCRow& row, const uint32_t& hitIndex) const { return mHitData[row.mHitNumberOffset + hitIndex]; }

GPUdi() calink GPUTPCTrackingData::FirstHitInBin(const GPUTPCRow& row, calink binIndex) const { return mFirstHitInBin[row.mFirstHitInBinOffset + binIndex]; }

GPUhdi() int32_t GPUTPCTrackingData::ClusterDataIndex(const GPUTPCRow& row, uint32_t hitIndex) const { return mClusterDataIndex[row.mHitNumberOffset + hitIndex]; }

GPUdi() void GPUTPCTrackingData::MaximizeHitWeight(const GPUTPCRow& row, uint32_t hitIndex, uint32_t weight)
{
  CAMath::AtomicMax(&mHitWeights[row.mHitNumberOffset + hitIndex], weight);
}

GPUdi() void GPUTPCTrackingData::SetHitWeight(const GPUTPCRow& row, uint32_t hitIndex, uint32_t weight)
{
  mHitWeights[row.mHitNumberOffset + hitIndex] = weight;
}

GPUdi() int32_t GPUTPCTrackingData::HitWeight(const GPUTPCRow& row, uint32_t hitIndex) const { return mHitWeights[row.mHitNumberOffset + hitIndex]; }
} // namespace o2::gpu

#endif // GPUTPCSECTORDATA_H
