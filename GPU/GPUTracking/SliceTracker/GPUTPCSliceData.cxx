// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceData.cxx
/// \author Matthias Kretz, Sergey Gorbunov, David Rohr

#include "GPUParam.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCHit.h"
#include "GPUTPCSliceData.h"
#include "GPUReconstruction.h"
#include "GPUProcessor.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCConvertImpl.h"
#include "GPUCommonMath.h"
#include <iostream>
#include <cstring>
#include "utils/vecpod.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_GPUCODE

void GPUTPCSliceData::InitializeRows(const MEM_CONSTANT(GPUParam) & p)
{
  // initialisation of rows
  for (int i = 0; i < GPUCA_ROW_COUNT + 1; ++i) {
    new (&mRows[i]) GPUTPCRow;
  }
  for (int i = 0; i < GPUCA_ROW_COUNT; ++i) {
    mRows[i].mX = p.tpcGeometry.Row2X(i);
    mRows[i].mMaxY = CAMath::Tan(p.DAlpha / 2.) * mRows[i].mX;
  }
}

void GPUTPCSliceData::SetClusterData(const GPUTPCClusterData* data, int nClusters, int clusterIdOffset)
{
  mClusterData = data;
  mNumberOfHits = nClusters;
  mClusterIdOffset = clusterIdOffset;
}

void GPUTPCSliceData::SetMaxData()
{
  int hitMemCount = GPUCA_ROW_COUNT * GPUCA_ROWALIGNMENT + mNumberOfHits;
  const unsigned int kVectorAlignment = 256;
  mNumberOfHitsPlusAlign = GPUProcessor::nextMultipleOf<(kVectorAlignment > GPUCA_ROWALIGNMENT ? kVectorAlignment : GPUCA_ROWALIGNMENT) / sizeof(int)>(hitMemCount);
}

unsigned int GPUTPCSliceData::GetGridSize()
{
  return (23 + GPUCA_ROWALIGNMENT / sizeof(int)) * GPUCA_ROW_COUNT + 4 * mNumberOfHits + 3;
}

void* GPUTPCSliceData::SetPointersInput(void* mem, bool idsOnGPU)
{
  const int firstHitInBinSize = GetGridSize();
  GPUProcessor::computePointerWithAlignment(mem, mHitData, mNumberOfHitsPlusAlign);
  GPUProcessor::computePointerWithAlignment(mem, mFirstHitInBin, firstHitInBinSize);
  if (idsOnGPU) {
    mem = SetPointersScratchHost(mem, false); // Hijack the allocation from SetPointersScratchHost
  }
  return mem;
}

void* GPUTPCSliceData::SetPointersScratch(const GPUConstantMem& cm, void* mem)
{
  GPUProcessor::computePointerWithAlignment(mem, mLinkUpData, mNumberOfHitsPlusAlign);
  GPUProcessor::computePointerWithAlignment(mem, mLinkDownData, mNumberOfHitsPlusAlign);
  GPUProcessor::computePointerWithAlignment(mem, mHitWeights, mNumberOfHitsPlusAlign + 16 / sizeof(*mHitWeights));
  size_t tmpMemMaxSize = CAMath::nextMultipleOf<4>(mNumberOfHits + 256) * (sizeof(float2) + sizeof(int) + sizeof(GPUTPCHit));
  int maxbiny, maxbinz;
  GetMaxNBins(&cm, maxbiny, maxbinz);
  size_t binMemoryMaxSize = (2 * (maxbiny * maxbinz + 3) + mNumberOfHits) * sizeof(calink) + GPUCA_ROW_COUNT * GPUCA_ROWALIGNMENT;
  tmpMemMaxSize += binMemoryMaxSize;
  GPUProcessor::computePointerWithAlignment(mem, mTmpMem, (tmpMemMaxSize + sizeof(*mTmpMem) - 1) / sizeof(*mTmpMem));
  return mem;
}

void* GPUTPCSliceData::SetPointersScratchHost(void* mem, bool idsOnGPU)
{
  if (!idsOnGPU) {
    GPUProcessor::computePointerWithAlignment(mem, mClusterDataIndex, mNumberOfHitsPlusAlign);
  }
  return mem;
}

void* GPUTPCSliceData::SetPointersRows(void* mem)
{
  GPUProcessor::computePointerWithAlignment(mem, mRows, GPUCA_ROW_COUNT + 1);
  return mem;
}

#endif

// calculates an approximation for 1/sqrt(x)
// Google for 0x5f3759df :)
static inline float fastInvSqrt(float _x)
{
  // the function calculates fast inverse sqrt
  union {
    float f;
    int i;
  } x = {_x};
  const float xhalf = 0.5f * x.f;
  x.i = 0x5f3759df - (x.i >> 1);
  x.f = x.f * (1.5f - xhalf * x.f * x.f);
  return x.f;
}

void GPUTPCSliceData::GetMaxNBins(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, int& maxY, int& maxZ)
{
  maxY = mRows[GPUCA_ROW_COUNT - 1].mMaxY * 2.f / GPUCA_MIN_BIN_SIZE + 1;
  maxZ = mem->param.continuousMaxTimeBin > 0 ? mem->calibObjects.fastTransform->convTimeToZinTimeFrame(0, 0, mem->param.continuousMaxTimeBin) + 50 : 300;
  maxZ = maxZ / GPUCA_MIN_BIN_SIZE + 1;
}

inline void GPUTPCSliceData::CreateGrid(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, GPUTPCRow* GPUrestrict() row, const float2* GPUrestrict() data, int ClusterDataHitNumberOffset)
{
  // grid creation
  if (row->NHits() <= 0) { // no hits or invalid data
    // grid coordinates don't matter, since there are no hits
    row->mGrid.CreateEmpty();
    return;
  }

  float yMin = 1.e6f;
  float yMax = -1.e6f;
  float zMin = 1.e6f;
  float zMax = -1.e6f;
  for (int i = ClusterDataHitNumberOffset; i < ClusterDataHitNumberOffset + row->mNHits; ++i) {
    const float y = data[i].x;
    const float z = data[i].y;
    if (yMax < y) {
      yMax = y;
    }
    if (yMin > y) {
      yMin = y;
    }
    if (zMax < z) {
      zMax = z;
    }
    if (zMin > z) {
      zMin = z;
    }
  }

  float dz = zMax - zMin;
  float tfFactor = 1.;
  if (dz > 270.) {
    tfFactor = dz / 250.;
    dz = 250.;
  }
  const float norm = fastInvSqrt(row->mNHits / tfFactor);
  float sy = CAMath::Min(CAMath::Max((yMax - yMin) * norm, GPUCA_MIN_BIN_SIZE), GPUCA_MAX_BIN_SIZE);
  float sz = CAMath::Min(CAMath::Max(dz * norm, GPUCA_MIN_BIN_SIZE), GPUCA_MAX_BIN_SIZE);
  int maxy, maxz;
  GetMaxNBins(mem, maxy, maxz);
  int ny = CAMath::Max(1, CAMath::Min<int>(maxy, (yMax - yMin) / sy + 1));
  int nz = CAMath::Max(1, CAMath::Min<int>(maxz, (zMax - zMin) / sz + 1));
  row->mGrid.Create(yMin, yMax, zMin, zMax, ny, nz);
}

inline int GPUTPCSliceData::PackHitData(GPUTPCRow* const GPUrestrict() row, const GPUTPCHit* GPUrestrict() binSortedHits)
{
  // hit data packing
  static const float maxVal = (((long long int)1 << CAMath::Min((size_t)24, sizeof(cahit) * 8)) - 1); // Stay within float precision in any case!
  static const float packingConstant = 1.f / (maxVal - 2.);
  const float y0 = row->mGrid.YMin();
  const float z0 = row->mGrid.ZMin();
  const float stepY = (row->mGrid.YMax() - y0) * packingConstant;
  const float stepZ = (row->mGrid.ZMax() - z0) * packingConstant;
  const float stepYi = 1.f / stepY;
  const float stepZi = 1.f / stepZ;

  row->mHy0 = y0;
  row->mHz0 = z0;
  row->mHstepY = stepY;
  row->mHstepZ = stepZ;
  row->mHstepYi = stepYi;
  row->mHstepZi = stepZi;

  for (int hitIndex = 0; hitIndex < row->mNHits; ++hitIndex) {
    // bin sorted index!
    const int globalHitIndex = row->mHitNumberOffset + hitIndex;
    const GPUTPCHit& hh = binSortedHits[hitIndex];
    const float xx = ((hh.Y() - y0) * stepYi) + .5;
    const float yy = ((hh.Z() - z0) * stepZi) + .5;
    if (xx < 0 || yy < 0 || xx > maxVal || yy > maxVal) {
      std::cout << "!!!! hit packing error!!! " << xx << " " << yy << " (" << maxVal << ")" << std::endl;
      return 1;
    }
    // HitData is bin sorted
    mHitData[globalHitIndex].x = (cahit)xx;
    mHitData[globalHitIndex].y = (cahit)yy;
  }
  return 0;
}

int GPUTPCSliceData::InitFromClusterData(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * GPUrestrict() mem, int iSlice)
{
  ////////////////////////////////////
  // 0. sort rows
  ////////////////////////////////////

  mMaxZ = 0.f;

  std::unique_ptr<float2[]> YZData_p(new float2[mNumberOfHits]);
  std::unique_ptr<int[]> tmpHitIndex_p(new int[mNumberOfHits]);
  float2* YZData = YZData_p.get();
  int* tmpHitIndex = tmpHitIndex_p.get();

  int RowOffset[GPUCA_ROW_COUNT];
  const unsigned int* NumberOfClustersInRow;
  unsigned int NumberOfClustersInRowA[GPUCA_ROW_COUNT];
  if (mem->param.earlyTpcTransform) {
    memset(NumberOfClustersInRowA, 0, GPUCA_ROW_COUNT * sizeof(NumberOfClustersInRowA[0]));
    for (int i = 0; i < mNumberOfHits; i++) {
      const int tmpRow = mClusterData[i].row;
      NumberOfClustersInRowA[tmpRow]++;
    }
    NumberOfClustersInRow = NumberOfClustersInRowA;
  } else {
    NumberOfClustersInRow = &mem->ioPtrs.clustersNative->nClusters[iSlice][0];
  }

  int tmpOffset = 0;
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    if ((long long int)NumberOfClustersInRow[i] >= ((long long int)1 << (sizeof(calink) * 8))) {
      GPUError("Too many clusters in row %d for row indexing (%d >= %lld), indexing insufficient", i, NumberOfClustersInRow[i], ((long long int)1 << (sizeof(calink) * 8)));
      return 1;
    }
    if (NumberOfClustersInRow[i] >= (1 << 24)) {
      GPUError("Too many clusters in row %d for hit id indexing (%d >= %d), indexing insufficient", i, NumberOfClustersInRow[i], 1 << 24);
      return 1;
    }
    RowOffset[i] = tmpOffset;
    tmpOffset += NumberOfClustersInRow[i];
  }

  if (mem->param.earlyTpcTransform) {
    int RowsFilled[GPUCA_ROW_COUNT];
    memset(RowsFilled, 0, GPUCA_ROW_COUNT * sizeof(int));
    for (int i = 0; i < mNumberOfHits; i++) {
      float2 tmp;
      tmp.x = mClusterData[i].y;
      tmp.y = mClusterData[i].z;
      if (fabsf(tmp.y) > mMaxZ) {
        mMaxZ = fabsf(tmp.y);
      }
      int tmpRow = mClusterData[i].row;
      int newIndex = RowOffset[tmpRow] + (RowsFilled[tmpRow])++;
      YZData[newIndex] = tmp;
      tmpHitIndex[newIndex] = i;
    }
  } else {
    size_t k = 0;
    for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
      for (unsigned int j = 0; j < NumberOfClustersInRow[i]; j++) {
        float2 tmp;
        float x;
        GPUTPCConvertImpl::convert(*mem, iSlice, i, mem->ioPtrs.clustersNative->clusters[iSlice][i][j].getPad(), mem->ioPtrs.clustersNative->clusters[iSlice][i][j].getTime(), x, tmp.x, tmp.y);
        tmpHitIndex[k] = k;
        YZData[k++] = tmp;
      }
    }
  }

  ////////////////////////////////////
  // 2. fill HitData and FirstHitInBin
  ////////////////////////////////////

  vecpod<GPUTPCHit> binSortedHits(mNumberOfHits + GPUCA_ROWALIGNMENT);

  unsigned int gridContentOffset = 0;
  unsigned int hitOffset = 0;

  unsigned int binCreationMemorySize = 103 * 2 + mNumberOfHits;
  vecpod<calink> binCreationMemory(binCreationMemorySize);

  for (int rowIndex = 0; rowIndex < GPUCA_ROW_COUNT; ++rowIndex) {
    GPUTPCRow& row = mRows[rowIndex];
    if (NumberOfClustersInRow[rowIndex] == 0) {
      row.mGrid.CreateEmpty();
      row.mNHits = 0;
      row.mFullSize = 0;
      row.mHitNumberOffset = 0;
      row.mFirstHitInBinOffset = 0;
      row.mHy0 = 0.f;
      row.mHz0 = 0.f;
      row.mHstepY = 1.f;
      row.mHstepZ = 1.f;
      row.mHstepYi = 1.f;
      row.mHstepZi = 1.f;
      continue;
    }
    row.mNHits = NumberOfClustersInRow[rowIndex];
    row.mHitNumberOffset = hitOffset;
    hitOffset += GPUProcessor::nextMultipleOf<GPUCA_ROWALIGNMENT / sizeof(calink)>(NumberOfClustersInRow[rowIndex]);

    row.mFirstHitInBinOffset = gridContentOffset;

    CreateGrid(mem, &row, YZData, RowOffset[rowIndex]);
    const GPUTPCGrid& grid = row.mGrid;
    const int numberOfBins = grid.N();
    if ((long long int)numberOfBins >= ((long long int)1 << (sizeof(calink) * 8))) {
      GPUError("Too many bins in row %d for grid (%d >= %lld), indexing insufficient", rowIndex, numberOfBins, ((long long int)1 << (sizeof(calink) * 8)));
      return 1;
    }
    const int nn = numberOfBins + grid.Ny() + 3;
    if (gridContentOffset + nn >= GetGridSize()) {
      GPUError("firstHitInBin overflow");
      return 1;
    }

    int binCreationMemorySizeNew = numberOfBins * 2 + 6 + row.mNHits + GPUCA_ROWALIGNMENT / sizeof(unsigned short) * (GPUCA_ROW_COUNT + 1) + 1;
    if (binCreationMemorySizeNew > binCreationMemorySize) {
      binCreationMemorySize = binCreationMemorySizeNew;
      binCreationMemory.resize(binCreationMemorySize);
    }

    calink* c = binCreationMemory.data(); // number of hits in all previous bins
    calink* bins = c + numberOfBins + 3;  // cache for the bin index for every hit in this row, 3 extra empty bins at the end!!!
    calink* filled = bins + row.mNHits;   // counts how many hits there are per bin

    for (unsigned int bin = 0; bin < row.mGrid.N() + 3; ++bin) {
      filled[bin] = 0; // initialize filled[] to 0
    }
    for (int hitIndex = 0; hitIndex < row.mNHits; ++hitIndex) {
      const int globalHitIndex = RowOffset[rowIndex] + hitIndex;
      const calink bin = row.mGrid.GetBin(YZData[globalHitIndex].x, YZData[globalHitIndex].y);

      bins[hitIndex] = bin;
      ++filled[bin];
    }

    calink n = 0;
    for (int bin = 0; bin < numberOfBins + 3; ++bin) {
      c[bin] = n;
      n += filled[bin];
    }

    for (int hitIndex = 0; hitIndex < row.mNHits; ++hitIndex) {
      const calink bin = bins[hitIndex];
      --filled[bin];
      const calink ind = c[bin] + filled[bin]; // generate an index for this hit that is >= c[bin] and < c[bin + 1]
      const int globalBinsortedIndex = row.mHitNumberOffset + ind;
      const int globalHitIndex = RowOffset[rowIndex] + hitIndex;

      // allows to find the global hit index / coordinates from a global bin sorted hit index
      mClusterDataIndex[globalBinsortedIndex] = tmpHitIndex[globalHitIndex];
      binSortedHits[ind].SetY(YZData[globalHitIndex].x);
      binSortedHits[ind].SetZ(YZData[globalHitIndex].y);
    }

    if (PackHitData(&row, binSortedHits.data())) {
      return 1;
    }

    for (int i = 0; i < numberOfBins; ++i) {
      mFirstHitInBin[row.mFirstHitInBinOffset + i] = c[i]; // global bin-sorted hit index
    }
    const calink a = c[numberOfBins];
    // grid.N is <= row.mNHits
    for (int i = numberOfBins; i < nn; ++i) {
      mFirstHitInBin[row.mFirstHitInBinOffset + i] = a;
    }

    row.mFullSize = nn;
    gridContentOffset += nn;

    // Make pointer aligned
    gridContentOffset = GPUProcessor::nextMultipleOf<GPUCA_ROWALIGNMENT / sizeof(calink)>(gridContentOffset);
  }

  return 0;
}
