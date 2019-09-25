// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceData.h
/// \author Matthias Kretz, Sergey Gorbunov, David Rohr

#ifndef GPUTPCSLICEDATA_H
#define GPUTPCSLICEDATA_H

#include "GPUTPCDef.h"
#include "GPUTPCRow.h"
#include "GPUCommonMath.h"
#include "GPUParam.h"
#include "GPUMemoryResource.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterData;
class GPUTPCHit;

MEM_CLASS_PRE()
class GPUTPCSliceData
{
 public:
  GPUTPCSliceData() : mFirstRow(0), mLastRow(GPUCA_ROW_COUNT - 1), mNumberOfHits(0), mNumberOfHitsPlusAlign(0), mClusterIdOffset(0), mMaxZ(0.f), mGPUTextureBase(nullptr), mRows(nullptr), mLinkUpData(nullptr), mLinkDownData(nullptr), mClusterData(nullptr) {}

#ifndef GPUCA_GPUCODE
  ~GPUTPCSliceData() CON_DEFAULT;
#endif //! GPUCA_GPUCODE

  MEM_CLASS_PRE2()
  void InitializeRows(const MEM_LG2(GPUParam) & parameters);

  /**
 * (Re)Create the data that is tuned for optimal performance of the algorithm from the cluster
 * data.
 */

  void SetMaxData();
  void SetClusterData(const GPUTPCClusterData* data, int nClusters, int clusterIdOffset);
  void* SetPointersInput(void* mem, bool idsOnGPU);
  void* SetPointersScratch(void* mem);
  void* SetPointersScratchHost(void* mem, bool idsOnGPU);
  void* SetPointersRows(void* mem);

  int InitFromClusterData();

  /**
 * Return the number of hits in this slice.
 */
  GPUhd() int NumberOfHits() const { return mNumberOfHits; }
  GPUhd() int NumberOfHitsPlusAlign() const { return mNumberOfHitsPlusAlign; }
  GPUhd() int ClusterIdOffset() const { return mClusterIdOffset; }

  /**
 * Access to the hit links.
 *
 * The links values give the hit index in the row above/below. Or -1 if there is no link.
 */
  MEM_TEMPLATE()
  GPUd() calink HitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex) const;
  MEM_TEMPLATE()
  GPUd() calink HitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex) const;

  MEM_TEMPLATE()
  GPUhdi() GPUglobalref() const cahit2* HitData(const MEM_TYPE(GPUTPCRow) & row) const { return &mHitData[row.mHitNumberOffset]; }
  GPUhd() GPUglobalref() const cahit2* HitData() const { return (mHitData); }
  MEM_TEMPLATE()
  GPUdi() GPUglobalref() const calink* HitLinkUpData(const MEM_TYPE(GPUTPCRow) & row) const { return &mLinkUpData[row.mHitNumberOffset]; }
  MEM_TEMPLATE()
  GPUdi() GPUglobalref() const calink* HitLinkDownData(const MEM_TYPE(GPUTPCRow) & row) const { return &mLinkDownData[row.mHitNumberOffset]; }
  MEM_TEMPLATE()
  GPUdi() GPUglobalref() const calink* FirstHitInBin(const MEM_TYPE(GPUTPCRow) & row) const { return &mFirstHitInBin[row.mFirstHitInBinOffset]; }

  MEM_TEMPLATE()
  GPUd() void SetHitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex, const calink& value);
  MEM_TEMPLATE()
  GPUd() void SetHitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex, const calink& value);

  /**
 * Return the y and z coordinate(s) of the given hit(s).
 */
  MEM_TEMPLATE()
  GPUd() cahit HitDataY(const MEM_TYPE(GPUTPCRow) & row, const unsigned int& hitIndex) const;
  MEM_TEMPLATE()
  GPUd() cahit HitDataZ(const MEM_TYPE(GPUTPCRow) & row, const unsigned int& hitIndex) const;
  MEM_TEMPLATE()
  GPUd() cahit2 HitData(const MEM_TYPE(GPUTPCRow) & row, const unsigned int& hitIndex) const;

  /**
 * For a given bin index, content tells how many hits there are in the preceding bins. This maps
 * directly to the hit index in the given row.
 *
 * \param binIndexes in the range 0 to row.Grid.N + row.Grid.Ny + 3.
 */
  MEM_TEMPLATE()
  GPUd() calink FirstHitInBin(const MEM_TYPE(GPUTPCRow) & row, calink binIndexes) const;

  /**
 * If the given weight is higher than what is currently stored replace with the new weight.
 */
  MEM_TEMPLATE()
  GPUd() void MaximizeHitWeight(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex, int weight);
  MEM_TEMPLATE()
  GPUd() void SetHitWeight(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex, int weight);

  /**
 * Return the maximal weight the given hit got from one tracklet
 */
  MEM_TEMPLATE()
  GPUd() int HitWeight(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex) const;

  /**
 * Returns the index in the original GPUTPCClusterData object of the given hit
 */
  MEM_TEMPLATE()
  GPUhd() int ClusterDataIndex(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex) const;

  /**
 * Return the row object for the given row index.
 */
  GPUhdi() GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & Row(int rowIndex) const { return mRows[rowIndex]; }
  GPUhdi() GPUglobalref() MEM_GLOBAL(GPUTPCRow) * Rows() const { return mRows; }

  GPUhdi() GPUglobalref() GPUAtomic(unsigned int) * HitWeights() const { return (mHitWeights); }

  GPUhdi() void SetGPUTextureBase(const void* val) { mGPUTextureBase = val; }
  GPUhdi() char* GPUTextureBase() const { return ((char*)mGPUTextureBase); }
  GPUhdi() char* GPUTextureBaseConst() const { return ((char*)mGPUTextureBase); }

#if !defined(__OPENCL__)
  GPUhi() const GPUTPCClusterData* ClusterData() const
  {
    return mClusterData;
  }
#endif

  float MaxZ() const
  {
    return mMaxZ;
  }

 private:
#ifndef GPUCA_GPUCODE
  GPUTPCSliceData& operator=(const GPUTPCSliceData&) CON_DELETE; // ROOT 5 tries to use this if it is not private
  GPUTPCSliceData(const GPUTPCSliceData&) CON_DELETE;            //
  void CreateGrid(GPUTPCRow* row, const float2* data, int ClusterDataHitNumberOffset);
  int PackHitData(GPUTPCRow* row, const GPUTPCHit* binSortedHits);
#endif

  int mFirstRow; // First non-empty row
  int mLastRow;  // Last non-empty row

  int mNumberOfHits; // the number of hits in this slice
  int mNumberOfHitsPlusAlign;
  int mClusterIdOffset;

  float mMaxZ;

  GPUglobalref() const void* mGPUTextureBase; // pointer to start of GPU texture

  GPUglobalref() MEM_GLOBAL(GPUTPCRow) * mRows; // The row objects needed for most accessor functions

  GPUglobalref() calink* mLinkUpData;    // hit index in the row above which is linked to the given (global) hit index
  GPUglobalref() calink* mLinkDownData;  // hit index in the row below which is linked to the given (global) hit index
  GPUglobalref() cahit2* mHitData;       // packed y,z coordinate of the given (global) hit index
  GPUglobalref() int* mClusterDataIndex; // see ClusterDataIndex()

  /*
 * The size of the array is row.Grid.N + row.Grid.Ny + 3. The row.Grid.Ny + 3 is an optimization
 * to remove the need for bounds checking. The last values are the same as the entry at [N - 1].
 */
  GPUglobalref() calink* mFirstHitInBin;                // see FirstHitInBin
  GPUglobalref() GPUAtomic(unsigned int) * mHitWeights; // the weight of the longest tracklet crossed the cluster
  GPUglobalref() const GPUTPCClusterData* mClusterData;
};

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() calink MEM_LG(GPUTPCSliceData)::HitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex) const { return mLinkUpData[row.mHitNumberOffset + hitIndex]; }

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() calink MEM_LG(GPUTPCSliceData)::HitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex) const { return mLinkDownData[row.mHitNumberOffset + hitIndex]; }

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() void MEM_LG(GPUTPCSliceData)::SetHitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex, const calink& value)
{
  mLinkUpData[row.mHitNumberOffset + hitIndex] = value;
}

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() void MEM_LG(GPUTPCSliceData)::SetHitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, const calink& hitIndex, const calink& value)
{
  mLinkDownData[row.mHitNumberOffset + hitIndex] = value;
}

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() cahit MEM_LG(GPUTPCSliceData)::HitDataY(const MEM_TYPE(GPUTPCRow) & row, const unsigned int& hitIndex) const { return mHitData[row.mHitNumberOffset + hitIndex].x; }

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() cahit MEM_LG(GPUTPCSliceData)::HitDataZ(const MEM_TYPE(GPUTPCRow) & row, const unsigned int& hitIndex) const { return mHitData[row.mHitNumberOffset + hitIndex].y; }

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() cahit2 MEM_LG(GPUTPCSliceData)::HitData(const MEM_TYPE(GPUTPCRow) & row, const unsigned int& hitIndex) const { return mHitData[row.mHitNumberOffset + hitIndex]; }

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() calink MEM_LG(GPUTPCSliceData)::FirstHitInBin(const MEM_TYPE(GPUTPCRow) & row, calink binIndexes) const { return mFirstHitInBin[row.mFirstHitInBinOffset + binIndexes]; }

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUhdi() int MEM_LG(GPUTPCSliceData)::ClusterDataIndex(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex) const { return mClusterDataIndex[row.mHitNumberOffset + hitIndex]; }

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() void MEM_LG(GPUTPCSliceData)::MaximizeHitWeight(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex, int weight)
{
  CAMath::AtomicMax(&mHitWeights[row.mHitNumberOffset + hitIndex], weight);
}

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() void MEM_LG(GPUTPCSliceData)::SetHitWeight(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex, int weight)
{
  mHitWeights[row.mHitNumberOffset + hitIndex] = weight;
}

MEM_CLASS_PRE()
MEM_TEMPLATE()
GPUdi() int MEM_LG(GPUTPCSliceData)::HitWeight(const MEM_TYPE(GPUTPCRow) & row, unsigned int hitIndex) const { return mHitWeights[row.mHitNumberOffset + hitIndex]; }
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCSLICEDATA_H
