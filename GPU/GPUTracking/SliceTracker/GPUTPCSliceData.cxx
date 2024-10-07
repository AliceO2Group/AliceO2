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

/// \file GPUTPCSliceData.cxx
/// \author Matthias Kretz, Sergey Gorbunov, David Rohr

#include "GPUParam.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCHit.h"
#include "GPUTPCSliceData.h"
#include "GPUProcessor.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCConvertImpl.h"
#include "GPUCommonMath.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include "utils/vecpod.h"
#include <iostream>
#include <cstring>
#include "GPUReconstruction.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_GPUCODE

void GPUTPCSliceData::InitializeRows(const MEM_CONSTANT(GPUParam) & p)
{
  // initialisation of rows
  for (int32_t i = 0; i < GPUCA_ROW_COUNT + 1; ++i) {
    new (&mRows[i]) GPUTPCRow;
  }
  for (int32_t i = 0; i < GPUCA_ROW_COUNT; ++i) {
    mRows[i].mX = p.tpcGeometry.Row2X(i);
    mRows[i].mMaxY = CAMath::Tan(p.par.dAlpha / 2.f) * mRows[i].mX;
  }
}

void GPUTPCSliceData::SetClusterData(const GPUTPCClusterData* data, int32_t nClusters, int32_t clusterIdOffset)
{
  mClusterData = data;
  mNumberOfHits = nClusters;
  mClusterIdOffset = clusterIdOffset;
}

void GPUTPCSliceData::SetMaxData()
{
  int32_t hitMemCount = GPUCA_ROW_COUNT * GPUCA_ROWALIGNMENT + mNumberOfHits;
  const uint32_t kVectorAlignment = 256;
  mNumberOfHitsPlusAlign = GPUProcessor::nextMultipleOf<(kVectorAlignment > GPUCA_ROWALIGNMENT ? kVectorAlignment : GPUCA_ROWALIGNMENT) / sizeof(int32_t)>(hitMemCount);
}

void* GPUTPCSliceData::SetPointersInput(void* mem, bool idsOnGPU, bool sliceDataOnGPU)
{
  if (sliceDataOnGPU) {
    return mem;
  }
  const int32_t firstHitInBinSize = GetGridSize(mNumberOfHits, GPUCA_ROW_COUNT) + GPUCA_ROW_COUNT * GPUCA_ROWALIGNMENT / sizeof(int32_t);
  GPUProcessor::computePointerWithAlignment(mem, mHitData, mNumberOfHitsPlusAlign);
  GPUProcessor::computePointerWithAlignment(mem, mFirstHitInBin, firstHitInBinSize);
  if (idsOnGPU) {
    mem = SetPointersClusterIds(mem, false); // Hijack the allocation from SetPointersClusterIds
  }
  return mem;
}

void* GPUTPCSliceData::SetPointersLinks(void* mem)
{
  GPUProcessor::computePointerWithAlignment(mem, mLinkUpData, mNumberOfHitsPlusAlign);
  GPUProcessor::computePointerWithAlignment(mem, mLinkDownData, mNumberOfHitsPlusAlign);
  return mem;
}

void* GPUTPCSliceData::SetPointersWeights(void* mem)
{
  GPUProcessor::computePointerWithAlignment(mem, mHitWeights, mNumberOfHitsPlusAlign + 16 / sizeof(*mHitWeights));
  return mem;
}

void* GPUTPCSliceData::SetPointersScratch(void* mem, bool idsOnGPU, bool sliceDataOnGPU)
{
  if (sliceDataOnGPU) {
    mem = SetPointersInput(mem, idsOnGPU, false);
  }
  return mem;
}

void* GPUTPCSliceData::SetPointersClusterIds(void* mem, bool idsOnGPU)
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

GPUd() void GPUTPCSliceData::GetMaxNBins(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, GPUTPCRow* GPUrestrict() row, int32_t& maxY, int32_t& maxZ)
{
  maxY = row->mMaxY * 2.f / GPUCA_MIN_BIN_SIZE + 1;
  maxZ = (mem->param.continuousMaxTimeBin > 0 ? (mem->calibObjects.fastTransformHelper->getCorrMap()->convTimeToZinTimeFrame(0, 0, mem->param.continuousMaxTimeBin)) : mem->param.tpcGeometry.TPCLength()) + 50;
  maxZ = maxZ / GPUCA_MIN_BIN_SIZE + 1;
}

GPUd() uint32_t GPUTPCSliceData::GetGridSize(uint32_t nHits, uint32_t nRows)
{
  return 128 * nRows + 4 * nHits;
}

GPUdi() void GPUTPCSliceData::CreateGrid(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, GPUTPCRow* GPUrestrict() row, float yMin, float yMax, float zMin, float zMax)
{
  float dz = zMax - zMin;
  float tfFactor = 1.f;
  if (dz > GPUTPCGeometry::TPCLength() + 20.f) {
    tfFactor = dz / GPUTPCGeometry::TPCLength();
    dz = GPUTPCGeometry::TPCLength();
  }
  const float norm = CAMath::InvSqrt(row->mNHits / tfFactor);
  float sy = CAMath::Min(CAMath::Max((yMax - yMin) * norm, GPUCA_MIN_BIN_SIZE), GPUCA_MAX_BIN_SIZE);
  float sz = CAMath::Min(CAMath::Max(dz * norm, GPUCA_MIN_BIN_SIZE), GPUCA_MAX_BIN_SIZE);
  int32_t maxy, maxz;
  GetMaxNBins(mem, row, maxy, maxz);
  int32_t ny = CAMath::Max(1, CAMath::Min<int32_t>(maxy, (yMax - yMin) / sy + 1));
  int32_t nz = CAMath::Max(1, CAMath::Min<int32_t>(maxz, (zMax - zMin) / sz + 1));
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

GPUdii() void GPUTPCSliceData::SetRowGridEmpty(GPUTPCRow& GPUrestrict() row)
{
  GPUAtomic(calink)* c = (GPUAtomic(calink)*)mFirstHitInBin + row.mFirstHitInBinOffset;
  row.mGrid.CreateEmpty();
  row.mNHits = 0;
  row.mHitNumberOffset = 0;
  row.mHy0 = 0.f;
  row.mHz0 = 0.f;
  row.mHstepY = 1.f;
  row.mHstepZ = 1.f;
  row.mHstepYi = 1.f;
  row.mHstepZi = 1.f;
  for (int32_t i = 0; i < 4; i++) {
    c[i] = 0;
  }
}

GPUdii() int32_t GPUTPCSliceData::InitFromClusterData(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * GPUrestrict() mem, int32_t iSlice, float* tmpMinMax)
{
#ifdef GPUCA_GPUCODE
  constexpr bool EarlyTransformWithoutClusterNative = false;
#else
  bool EarlyTransformWithoutClusterNative = mem->param.par.earlyTpcTransform && mem->ioPtrs.clustersNative == nullptr;
#endif
  int32_t* tmpHitIndex = nullptr;
  const uint32_t* NumberOfClustersInRow = nullptr;
  const uint32_t* RowOffsets = nullptr;

#ifndef GPUCA_GPUCODE
  vecpod<float2> YZData(mNumberOfHits);
  vecpod<calink> binMemory(mNumberOfHits);
  uint32_t RowOffsetsA[GPUCA_ROW_COUNT];
  uint32_t NumberOfClustersInRowA[GPUCA_ROW_COUNT];

  vecpod<int32_t> tmpHitIndexA;
  if (EarlyTransformWithoutClusterNative) { // Implies mem->param.par.earlyTpcTransform but no ClusterNative present
    NumberOfClustersInRow = NumberOfClustersInRowA;
    RowOffsets = RowOffsetsA;
    tmpHitIndexA.resize(mNumberOfHits);
    tmpHitIndex = tmpHitIndexA.data();

    memset(NumberOfClustersInRowA, 0, GPUCA_ROW_COUNT * sizeof(NumberOfClustersInRowA[0]));
    for (int32_t i = 0; i < mNumberOfHits; i++) {
      const int32_t tmpRow = mClusterData[i].row;
      NumberOfClustersInRowA[tmpRow]++;
    }
    int32_t tmpOffset = 0;
    for (int32_t i = 0; i < GPUCA_ROW_COUNT; i++) {
      RowOffsetsA[i] = tmpOffset;
      tmpOffset += NumberOfClustersInRow[i];
    }
    int32_t RowsFilled[GPUCA_ROW_COUNT];
    memset(RowsFilled, 0, GPUCA_ROW_COUNT * sizeof(int32_t));
    for (int32_t i = 0; i < mNumberOfHits; i++) {
      float2 tmp;
      tmp.x = mClusterData[i].y;
      tmp.y = mClusterData[i].z;
      int32_t tmpRow = mClusterData[i].row;
      int32_t newIndex = RowOffsetsA[tmpRow] + (RowsFilled[tmpRow])++;
      YZData[newIndex] = tmp;
      tmpHitIndex[newIndex] = i;
    }
  } // Other cases below in loop over rows
#else
  float2* YZData = (float2*)mLinkUpData; // TODO: we can do this as well on the CPU, just must make sure that CPU has the scratch memory
  calink* binMemory = (calink*)mHitWeights;
  static_assert(sizeof(*YZData) <= (sizeof(*mLinkUpData) + sizeof(*mLinkDownData)), "Cannot reuse memory");
  static_assert(sizeof(*binMemory) <= sizeof(*mHitWeights), "Cannot reuse memory");
#endif

  for (int32_t rowIndex = iBlock; rowIndex < GPUCA_ROW_COUNT; rowIndex += nBlocks) {
    float yMin = 1.e6f;
    float yMax = -1.e6f;
    float zMin = 1.e6f;
    float zMax = -1.e6f;

    const uint32_t NumberOfClusters = EarlyTransformWithoutClusterNative ? NumberOfClustersInRow[rowIndex] : mem->ioPtrs.clustersNative->nClusters[iSlice][rowIndex];
    const uint32_t RowOffset = EarlyTransformWithoutClusterNative ? RowOffsets[rowIndex] : (mem->ioPtrs.clustersNative->clusterOffset[iSlice][rowIndex] - mem->ioPtrs.clustersNative->clusterOffset[iSlice][0]);
    CONSTEXPR const uint32_t maxN = 1u << (sizeof(calink) < 3 ? (sizeof(calink) * 8) : 24);
    GPUTPCRow& row = mRows[rowIndex];
    if (iThread == 0) {
      row.mFirstHitInBinOffset = CAMath::nextMultipleOf<GPUCA_ROWALIGNMENT / sizeof(calink)>(GetGridSize(RowOffset, rowIndex) + rowIndex * GPUCA_ROWALIGNMENT / sizeof(int32_t));
    }
    if (NumberOfClusters >= maxN) {
      if (iThread == 0) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_HITINROW_OVERFLOW, iSlice * 1000 + rowIndex, NumberOfClusters, maxN);
        SetRowGridEmpty(row);
      }
      continue;
    }

    if (iThread == 0) {
      tmpMinMax[0] = yMin;
      tmpMinMax[1] = yMax;
      tmpMinMax[2] = zMin;
      tmpMinMax[3] = zMax;
    }
    GPUbarrier();
    GPUAtomic(calink)* c = (GPUAtomic(calink)*)mFirstHitInBin + row.mFirstHitInBinOffset;
    if (NumberOfClusters == 0) {
      if (iThread == 0) {
        SetRowGridEmpty(row);
      }
      continue;
    }

    if (EarlyTransformWithoutClusterNative) {
      for (uint32_t i = iThread; i < NumberOfClusters; i += nThreads) {
        UpdateMinMaxYZ(yMin, yMax, zMin, zMax, YZData[RowOffset + i].x, YZData[RowOffset + i].y);
      }
    } else {
      if (mem->param.par.earlyTpcTransform) { // Early transform case with ClusterNative present
        for (uint32_t i = iThread; i < NumberOfClusters; i += nThreads) {
          float2 tmp;
          tmp.x = mClusterData[RowOffset + i].y;
          tmp.y = mClusterData[RowOffset + i].z;
          UpdateMinMaxYZ(yMin, yMax, zMin, zMax, tmp.x, tmp.y);
          YZData[RowOffset + i] = tmp;
        }
      } else {
        for (uint32_t i = iThread; i < NumberOfClusters; i += nThreads) {
          float x, y, z;
          GPUTPCConvertImpl::convert(*mem, iSlice, rowIndex, mem->ioPtrs.clustersNative->clusters[iSlice][rowIndex][i].getPad(), mem->ioPtrs.clustersNative->clusters[iSlice][rowIndex][i].getTime(), x, y, z);
          UpdateMinMaxYZ(yMin, yMax, zMin, zMax, y, z);
          YZData[RowOffset + i] = CAMath::MakeFloat2(y, z);
        }
      }
    }

    if (iThread == 0) {
      row.mNHits = NumberOfClusters;
      row.mHitNumberOffset = CAMath::nextMultipleOf<GPUCA_ROWALIGNMENT / sizeof(calink)>(RowOffset + rowIndex * GPUCA_ROWALIGNMENT / sizeof(calink));
    }

#ifdef GPUCA_HAVE_ATOMIC_MINMAX_FLOAT
    CAMath::AtomicMinShared(&tmpMinMax[0], yMin);
    CAMath::AtomicMaxShared(&tmpMinMax[1], yMax);
    CAMath::AtomicMinShared(&tmpMinMax[2], zMin);
    CAMath::AtomicMaxShared(&tmpMinMax[3], zMax);
#else
    for (int32_t i = 0; i < nThreads; i++) {
      GPUbarrier();
      if (iThread == i) {
        if (tmpMinMax[0] > yMin) {
          tmpMinMax[0] = yMin;
        }
        if (tmpMinMax[1] < yMax) {
          tmpMinMax[1] = yMax;
        }
        if (tmpMinMax[2] > zMin) {
          tmpMinMax[2] = zMin;
        }
        if (tmpMinMax[3] < zMax) {
          tmpMinMax[3] = zMax;
        }
      }
    }
#endif
    GPUbarrier();
    if (iThread == 0) {
      CreateGrid(mem, &row, tmpMinMax[0], tmpMinMax[1], tmpMinMax[2], tmpMinMax[3]);
    }
    GPUbarrier();
    const GPUTPCGrid& grid = row.mGrid;
    const int32_t numberOfBins = grid.N();
    CONSTEXPR const int32_t maxBins = sizeof(calink) < 4 ? (int32_t)(1ul << (sizeof(calink) * 8)) : 0x7FFFFFFF; // NOLINT: false warning
    if (sizeof(calink) < 4 && numberOfBins >= maxBins) {
      if (iThread == 0) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_BIN_OVERFLOW, iSlice * 1000 + rowIndex, numberOfBins, maxBins);
        SetRowGridEmpty(row);
      }
      continue;
    }
    const uint32_t nn = numberOfBins + grid.Ny() + 3;
    const uint32_t maxnn = GetGridSize(NumberOfClusters, 1);
    if (nn >= maxnn) {
      if (iThread == 0) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_FIRSTHITINBIN_OVERFLOW, iSlice, nn, maxnn);
        SetRowGridEmpty(row);
      }
      continue;
    }

    calink* bins = &binMemory[RowOffset]; // Reuse mLinkUpData memory as temporary memory

    for (int32_t bin = iThread; bin < numberOfBins; bin += nThreads) {
      c[bin] = 0; // initialize to 0
    }
    GPUbarrier();
    for (int32_t hitIndex = iThread; hitIndex < row.mNHits; hitIndex += nThreads) {
      const int32_t globalHitIndex = RowOffset + hitIndex;
      const calink bin = row.mGrid.GetBin(YZData[globalHitIndex].x, YZData[globalHitIndex].y);

      bins[hitIndex] = bin;
      CAMath::AtomicAdd(&c[bin], 1u);
    }
    GPUbarrier();

    if (iThread == 0) {
      calink n = 0;
      for (int32_t bin = 0; bin < numberOfBins; ++bin) { // TODO: Parallelize
        n += c[bin];
        c[bin] = n;
      }
      for (uint32_t bin = numberOfBins; bin < nn; bin++) {
        c[bin] = n;
      }
    }
    GPUbarrier();

    constexpr float maxVal = (((int64_t)1 << (sizeof(cahit) < 3 ? sizeof(cahit) * 8 : 24)) - 1); // Stay within float precision in any case!
    constexpr float packingConstant = 1.f / (maxVal - 2.f);
    const float y0 = row.mGrid.YMin();
    const float z0 = row.mGrid.ZMin();
    const float stepY = (row.mGrid.YMax() - y0) * packingConstant;
    const float stepZ = (row.mGrid.ZMax() - z0) * packingConstant;
    const float stepYi = 1.f / stepY;
    const float stepZi = 1.f / stepZ;

    if (iThread == 0) {
      row.mHy0 = y0;
      row.mHz0 = z0;
      row.mHstepY = stepY;
      row.mHstepZ = stepZ;
      row.mHstepYi = stepYi;
      row.mHstepZi = stepZi;
    }

    GPUbarrier();

    for (int32_t hitIndex = iThread; hitIndex < row.mNHits; hitIndex += nThreads) {
      const calink bin = bins[hitIndex];
      const calink ind = CAMath::AtomicAdd(&c[bin], (calink)-1) - 1; // generate an index for this hit that is >= c[bin] and < c[bin + 1]
      const int32_t globalBinsortedIndex = row.mHitNumberOffset + ind;
      const int32_t globalHitIndex = RowOffset + hitIndex;

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

    GPUbarrier();

    if (iThread == 0 && !mem->param.par.continuousTracking) {
      const float maxAbsZ = CAMath::Max(CAMath::Abs(tmpMinMax[2]), CAMath::Abs(tmpMinMax[3]));
      if (maxAbsZ > 300) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_Z_OVERFLOW, iSlice, (uint32_t)maxAbsZ);
        SetRowGridEmpty(row);
        continue;
      }
    }
  }

  return 0;
}
