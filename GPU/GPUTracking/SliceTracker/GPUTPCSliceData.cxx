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

#ifndef __OPENCL__
#include "utils/vecpod.h"
#include <iostream>
#include <cstring>
#endif

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

void* GPUTPCSliceData::SetPointersInput(void* mem, bool idsOnGPU)
{
  const int firstHitInBinSize = GetGridSize(mNumberOfHits, GPUCA_ROW_COUNT) + GPUCA_ROW_COUNT * GPUCA_ROWALIGNMENT / sizeof(int);
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

GPUd() void GPUTPCSliceData::GetMaxNBins(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, GPUTPCRow* GPUrestrict() row, int& maxY, int& maxZ)
{
  maxY = row->mMaxY * 2.f / GPUCA_MIN_BIN_SIZE + 1;
  maxZ = mem->param.continuousMaxTimeBin > 0 ? mem->calibObjects.fastTransform->convTimeToZinTimeFrame(0, 0, mem->param.continuousMaxTimeBin) + 50 : 300;
  maxZ = maxZ / GPUCA_MIN_BIN_SIZE + 1;
}

GPUd() unsigned int GPUTPCSliceData::GetGridSize(unsigned int nHits, unsigned int nRows)
{
  return 26 * nRows + 4 * nHits;
}

GPUdi() void GPUTPCSliceData::CreateGrid(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, GPUTPCRow* GPUrestrict() row, const float2* GPUrestrict() data, int ClusterDataHitNumberOffset, float yMin, float yMax, float zMin, float zMax)
{
  float dz = zMax - zMin;
  float tfFactor = 1.;
  if (dz > 270.) {
    tfFactor = dz / 250.;
    dz = 250.;
  }
  const float norm = CAMath::FastInvSqrt(row->mNHits / tfFactor);
  float sy = CAMath::Min(CAMath::Max((yMax - yMin) * norm, GPUCA_MIN_BIN_SIZE), GPUCA_MAX_BIN_SIZE);
  float sz = CAMath::Min(CAMath::Max(dz * norm, GPUCA_MIN_BIN_SIZE), GPUCA_MAX_BIN_SIZE);
  int maxy, maxz;
  GetMaxNBins(mem, row, maxy, maxz);
  int ny = CAMath::Max(1, CAMath::Min<int>(maxy, (yMax - yMin) / sy + 1));
  int nz = CAMath::Max(1, CAMath::Min<int>(maxz, (zMax - zMin) / sz + 1));
  row->mGrid.Create(yMin, yMax, zMin, zMax, ny, nz);
}

GPUdi() static void UpdateMinMaxYZ(float& yMin, float& yMax, float& zMin, float& zMax, float y, float z)
{
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

GPUd() int GPUTPCSliceData::InitFromClusterData(int nBlocks, int nThreads, int iBlock, int iThread, GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * GPUrestrict() mem, int iSlice)
{
#ifdef GPUCA_GPUCODE
  constexpr bool EarlyTransformWithoutClusterNative = false;
  if (mem->ioPtrs.clustersNative == nullptr) {
    GPUError("Cluster Native Access Structure missing");
    return 1;
  }
#else
  bool EarlyTransformWithoutClusterNative = mem->param.earlyTpcTransform && mem->ioPtrs.clustersNative == nullptr;
#endif
  int* tmpHitIndex = nullptr;
  const unsigned int* NumberOfClustersInRow = nullptr;
  const unsigned int* RowOffsets = nullptr;

#ifndef GPUCA_GPUCODE
  std::unique_ptr<float2[]> YZData_p(new float2[mNumberOfHits]);
  float2* YZData = YZData_p.get();
  unsigned int RowOffsetsA[GPUCA_ROW_COUNT];
  unsigned int NumberOfClustersInRowA[GPUCA_ROW_COUNT];

  std::unique_ptr<int[]> tmpHitIndex_p(new int[mNumberOfHits]);
  if (EarlyTransformWithoutClusterNative) { // Implies mem->param.earlyTpcTransform but no ClusterNative present
    NumberOfClustersInRow = NumberOfClustersInRowA;
    RowOffsets = RowOffsetsA;
    tmpHitIndex = tmpHitIndex_p.get();

    memset(NumberOfClustersInRowA, 0, GPUCA_ROW_COUNT * sizeof(NumberOfClustersInRowA[0]));
    for (int i = 0; i < mNumberOfHits; i++) {
      const int tmpRow = mClusterData[i].row;
      NumberOfClustersInRowA[tmpRow]++;
    }
    int tmpOffset = 0;
    for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
      RowOffsetsA[i] = tmpOffset;
      tmpOffset += NumberOfClustersInRow[i];
    }
    int RowsFilled[GPUCA_ROW_COUNT];
    memset(RowsFilled, 0, GPUCA_ROW_COUNT * sizeof(int));
    for (int i = 0; i < mNumberOfHits; i++) {
      float2 tmp;
      tmp.x = mClusterData[i].y;
      tmp.y = mClusterData[i].z;
      int tmpRow = mClusterData[i].row;
      int newIndex = RowOffsetsA[tmpRow] + (RowsFilled[tmpRow])++;
      YZData[newIndex] = tmp;
      tmpHitIndex[newIndex] = i;
    }
  } // Other cases below in loop over rows
#else
  float2* YZData = (float2*)mLinkUpData;
  static_assert(sizeof(*YZData) <= (sizeof(*mLinkUpData) + sizeof(*mLinkDownData)), "Cannot reuse memory");
#endif

  for (int rowIndex = 0; rowIndex < GPUCA_ROW_COUNT; ++rowIndex) {
    float yMin = 1.e6f;
    float yMax = -1.e6f;
    float zMin = 1.e6f;
    float zMax = -1.e6f;

    const unsigned int NumberOfClusters = EarlyTransformWithoutClusterNative ? NumberOfClustersInRow[rowIndex] : mem->ioPtrs.clustersNative->nClusters[iSlice][rowIndex];
    const unsigned int RowOffset = EarlyTransformWithoutClusterNative ? RowOffsets[rowIndex] : (mem->ioPtrs.clustersNative->clusterOffset[iSlice][rowIndex] - mem->ioPtrs.clustersNative->clusterOffset[iSlice][0]);
    if ((long long int)NumberOfClusters >= ((long long int)1 << (sizeof(calink) * 8))) {
      GPUError("Too many clusters in row %d for row indexing (%d >= %lld), indexing insufficient", rowIndex, NumberOfClusters, ((long long int)1 << (sizeof(calink) * 8)));
      return 1;
    }
    if (NumberOfClusters >= (1 << 24)) {
      GPUError("Too many clusters in row %d for hit id indexing (%d >= %d), indexing insufficient", rowIndex, NumberOfClusters, 1 << 24);
      return 1;
    }

    GPUTPCRow& row = mRows[rowIndex];
    if (NumberOfClusters == 0) {
      row.mGrid.CreateEmpty();
      row.mNHits = 0;
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

    if (EarlyTransformWithoutClusterNative) {
      for (unsigned int i = 0; i < NumberOfClusters; i++) {
        UpdateMinMaxYZ(yMin, yMax, zMin, zMax, YZData[RowOffset - i].x, YZData[RowOffset - i].y);
      }
    } else {
      if (mem->param.earlyTpcTransform) { // Early transform case with ClusterNative present
        for (unsigned int i = 0; i < NumberOfClusters; i++) {
          float2 tmp;
          tmp.x = mClusterData[RowOffset + i].y;
          tmp.y = mClusterData[RowOffset + i].z;
          UpdateMinMaxYZ(yMin, yMax, zMin, zMax, tmp.x, tmp.y);
          YZData[RowOffset + i] = tmp;
        }
      } else {
        for (unsigned int i = 0; i < NumberOfClusters; i++) {
          float x, y, z;
          GPUTPCConvertImpl::convert(*mem, iSlice, rowIndex, mem->ioPtrs.clustersNative->clusters[iSlice][rowIndex][i].getPad(), mem->ioPtrs.clustersNative->clusters[iSlice][rowIndex][i].getTime(), x, y, z);
          UpdateMinMaxYZ(yMin, yMax, zMin, zMax, y, z);
          YZData[RowOffset + i] = CAMath::MakeFloat2(y, z);
        }
      }
    }

    row.mNHits = NumberOfClusters;
    row.mHitNumberOffset = CAMath::nextMultipleOf<GPUCA_ROWALIGNMENT / sizeof(calink)>(RowOffset + rowIndex * GPUCA_ROWALIGNMENT / sizeof(calink));
    row.mFirstHitInBinOffset = CAMath::nextMultipleOf<GPUCA_ROWALIGNMENT / sizeof(calink)>(GetGridSize(RowOffset, rowIndex) + rowIndex * GPUCA_ROWALIGNMENT / sizeof(int));

    CreateGrid(mem, &row, YZData, RowOffset, yMin, yMax, zMin, zMax);
    const GPUTPCGrid& grid = row.mGrid;
    const int numberOfBins = grid.N();
    if ((long long int)numberOfBins >= ((long long int)1 << (sizeof(calink) * 8))) {
      GPUError("Too many bins in row %d for grid (%d >= %lld), indexing insufficient", rowIndex, numberOfBins, ((long long int)1 << (sizeof(calink) * 8)));
      return 1;
    }
    const unsigned int nn = numberOfBins + grid.Ny() + 3;
    if (nn >= GetGridSize(NumberOfClusters, 1)) {
      GPUError("firstHitInBin overflow");
      return 1;
    }

    calink* c = mFirstHitInBin + row.mFirstHitInBinOffset; // number of hits in all previous bins
    calink* bins = (calink*)mHitWeights + RowOffset;       // Reuse mLinkUpData memory as temporary memory
    static_assert(sizeof(*bins) <= sizeof(*mHitWeights), "Cannot reuse memory");

    for (int bin = 0; bin < numberOfBins; ++bin) {
      c[bin] = 0; // initialize filled[] to 0
    }
    for (int hitIndex = 0; hitIndex < row.mNHits; ++hitIndex) {
      const int globalHitIndex = RowOffset + hitIndex;
      const calink bin = row.mGrid.GetBin(YZData[globalHitIndex].x, YZData[globalHitIndex].y);

      bins[hitIndex] = bin;
      ++c[bin];
    }

    calink n = 0;
    for (int bin = 0; bin < numberOfBins; ++bin) {
      n += c[bin];
      c[bin] = n;
    }
    for (unsigned int bin = numberOfBins; bin < nn; bin++) {
      c[bin] = n;
    }

    constexpr float maxVal = (((long long int)1 << (sizeof(cahit) < 3 ? sizeof(cahit) * 8 : 24)) - 1); // Stay within float precision in any case!
    constexpr float packingConstant = 1.f / (maxVal - 2.);
    const float y0 = row.mGrid.YMin();
    const float z0 = row.mGrid.ZMin();
    const float stepY = (row.mGrid.YMax() - y0) * packingConstant;
    const float stepZ = (row.mGrid.ZMax() - z0) * packingConstant;
    const float stepYi = 1.f / stepY;
    const float stepZi = 1.f / stepZ;

    row.mHy0 = y0;
    row.mHz0 = z0;
    row.mHstepY = stepY;
    row.mHstepZ = stepZ;
    row.mHstepYi = stepYi;
    row.mHstepZi = stepZi;

    for (int hitIndex = 0; hitIndex < row.mNHits; ++hitIndex) {
      const calink bin = bins[hitIndex];
      const calink ind = --c[bin]; // generate an index for this hit that is >= c[bin] and < c[bin + 1]
      const int globalBinsortedIndex = row.mHitNumberOffset + ind;
      const int globalHitIndex = RowOffset + hitIndex;

      // allows to find the global hit index / coordinates from a global bin sorted hit index
      mClusterDataIndex[globalBinsortedIndex] = EarlyTransformWithoutClusterNative ? tmpHitIndex[globalHitIndex] : (RowOffset + hitIndex);

      const float xx = ((YZData[globalHitIndex].x - y0) * stepYi) + .5;
      const float yy = ((YZData[globalHitIndex].y - z0) * stepZi) + .5;
#if !defined(GPUCA_GPUCODE) && !defined(NDEBUG)
      if (xx < 0 || yy < 0 || xx > maxVal || yy > maxVal) {
        std::cout << "!!!! hit packing error!!! " << xx << " " << yy << " (" << maxVal << ")" << std::endl;
        return 1;
      }
#endif
      // HitData is bin sorted
      mHitData[globalBinsortedIndex].x = (cahit)xx;
      mHitData[globalBinsortedIndex].y = (cahit)yy;
    }

    const float maxAbsZ = CAMath::Max(CAMath::Abs(zMin), CAMath::Abs(zMax));
    if (maxAbsZ > 300 && !mem->param.ContinuousTracking) {
      GPUError("Need to set continuous tracking mode for data outside of the TPC volume!"); // TODO: Set GPU error code
      return 1;
    }
  }

  return 0;
}
