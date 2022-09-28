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
  for (int i = 0; i < GPUCA_ROW_COUNT + 1; ++i) {
    new (&mRows[i]) GPUTPCRow;
  }
  for (int i = 0; i < GPUCA_ROW_COUNT; ++i) {
    mRows[i].mX = p.tpcGeometry.Row2X(i);
    mRows[i].mMaxY = CAMath::Tan(p.par.dAlpha / 2.) * mRows[i].mX;
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

void* GPUTPCSliceData::SetPointersInput(void* mem, bool idsOnGPU, bool sliceDataOnGPU)
{
  if (sliceDataOnGPU) {
    return mem;
  }
  const int firstHitInBinSize = GetGridSize(mNumberOfHits, GPUCA_ROW_COUNT) + GPUCA_ROW_COUNT * GPUCA_ROWALIGNMENT / sizeof(int);
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

GPUd() void GPUTPCSliceData::GetMaxNBins(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, GPUTPCRow* GPUrestrict() row, int& maxY, int& maxZ)
{
  maxY = row->mMaxY * 2.f / GPUCA_MIN_BIN_SIZE + 1;
  maxZ = mem->param.par.continuousMaxTimeBin > 0 ? mem->calibObjects.fastTransformHelper->getCorrMap()->convTimeToZinTimeFrame(0, 0, mem->param.par.continuousMaxTimeBin) + 50 : 300;
  maxZ = maxZ / GPUCA_MIN_BIN_SIZE + 1;
}

GPUd() unsigned int GPUTPCSliceData::GetGridSize(unsigned int nHits, unsigned int nRows)
{
  return 128 * nRows + 4 * nHits;
}

GPUdi() void GPUTPCSliceData::CreateGrid(GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mem, GPUTPCRow* GPUrestrict() row, float yMin, float yMax, float zMin, float zMax)
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

GPUdii() int GPUTPCSliceData::InitFromClusterData(int nBlocks, int nThreads, int iBlock, int iThread, GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * GPUrestrict() mem, int iSlice, float* tmpMinMax)
{
#ifdef GPUCA_GPUCODE
  constexpr bool EarlyTransformWithoutClusterNative = false;
#else
  bool EarlyTransformWithoutClusterNative = mem->param.par.earlyTpcTransform && mem->ioPtrs.clustersNative == nullptr;
#endif
  int* tmpHitIndex = nullptr;
  const unsigned int* NumberOfClustersInRow = nullptr;
  const unsigned int* RowOffsets = nullptr;

#ifndef GPUCA_GPUCODE
  vecpod<float2> YZData(mNumberOfHits);
  vecpod<calink> binMemory(mNumberOfHits);
  unsigned int RowOffsetsA[GPUCA_ROW_COUNT];
  unsigned int NumberOfClustersInRowA[GPUCA_ROW_COUNT];

  vecpod<int> tmpHitIndexA;
  if (EarlyTransformWithoutClusterNative) { // Implies mem->param.par.earlyTpcTransform but no ClusterNative present
    NumberOfClustersInRow = NumberOfClustersInRowA;
    RowOffsets = RowOffsetsA;
    tmpHitIndexA.resize(mNumberOfHits);
    tmpHitIndex = tmpHitIndexA.data();

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
  float2* YZData = (float2*)mLinkUpData; // TODO: we can do this as well on the CPU, just must make sure that CPU has the scratch memory
  calink* binMemory = (calink*)mHitWeights;
  static_assert(sizeof(*YZData) <= (sizeof(*mLinkUpData) + sizeof(*mLinkDownData)), "Cannot reuse memory");
  static_assert(sizeof(*binMemory) <= sizeof(*mHitWeights), "Cannot reuse memory");
#endif

  for (int rowIndex = iBlock; rowIndex < GPUCA_ROW_COUNT; rowIndex += nBlocks) {
    float yMin = 1.e6f;
    float yMax = -1.e6f;
    float zMin = 1.e6f;
    float zMax = -1.e6f;

    const unsigned int NumberOfClusters = EarlyTransformWithoutClusterNative ? NumberOfClustersInRow[rowIndex] : mem->ioPtrs.clustersNative->nClusters[iSlice][rowIndex];
    const unsigned int RowOffset = EarlyTransformWithoutClusterNative ? RowOffsets[rowIndex] : (mem->ioPtrs.clustersNative->clusterOffset[iSlice][rowIndex] - mem->ioPtrs.clustersNative->clusterOffset[iSlice][0]);
    CONSTEXPR unsigned int maxN = 1u << (sizeof(calink) < 3 ? (sizeof(calink) * 8) : 24);
    if (NumberOfClusters >= maxN) {
      if (iThread == 0) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_HITINROW_OVERFLOW, iSlice * 1000 + rowIndex, NumberOfClusters, maxN);
      }
      return 1;
    }

    GPUTPCRow& row = mRows[rowIndex];
    if (iThread == 0) {
      tmpMinMax[0] = yMin;
      tmpMinMax[1] = yMax;
      tmpMinMax[2] = zMin;
      tmpMinMax[3] = zMax;
      row.mFirstHitInBinOffset = CAMath::nextMultipleOf<GPUCA_ROWALIGNMENT / sizeof(calink)>(GetGridSize(RowOffset, rowIndex) + rowIndex * GPUCA_ROWALIGNMENT / sizeof(int));
    }
    GPUbarrier();
    GPUAtomic(calink)* c = (GPUAtomic(calink)*)mFirstHitInBin + row.mFirstHitInBinOffset;
    if (NumberOfClusters == 0) {
      if (iThread == 0) {
        row.mGrid.CreateEmpty();
        row.mNHits = 0;
        row.mHitNumberOffset = 0;
        row.mHy0 = 0.f;
        row.mHz0 = 0.f;
        row.mHstepY = 1.f;
        row.mHstepZ = 1.f;
        row.mHstepYi = 1.f;
        row.mHstepZi = 1.f;
        for (int i = 0; i < 4; i++) {
          c[i] = 0;
        }
      }
      continue;
    }

    if (EarlyTransformWithoutClusterNative) {
      for (unsigned int i = iThread; i < NumberOfClusters; i += nThreads) {
        UpdateMinMaxYZ(yMin, yMax, zMin, zMax, YZData[RowOffset + i].x, YZData[RowOffset + i].y);
      }
    } else {
      if (mem->param.par.earlyTpcTransform) { // Early transform case with ClusterNative present
        for (unsigned int i = iThread; i < NumberOfClusters; i += nThreads) {
          float2 tmp;
          tmp.x = mClusterData[RowOffset + i].y;
          tmp.y = mClusterData[RowOffset + i].z;
          UpdateMinMaxYZ(yMin, yMax, zMin, zMax, tmp.x, tmp.y);
          YZData[RowOffset + i] = tmp;
        }
      } else {
        for (unsigned int i = iThread; i < NumberOfClusters; i += nThreads) {
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
    for (int i = 0; i < nThreads; i++) {
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
    const int numberOfBins = grid.N();
    CONSTEXPR int maxBins = sizeof(calink) < 4 ? (int)(1ul << (sizeof(calink) * 8)) : 0x7FFFFFFF; // NOLINT: false warning
    if (sizeof(calink) < 4 && numberOfBins >= maxBins) {
      if (iThread == 0) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_BIN_OVERFLOW, iSlice * 1000 + rowIndex, numberOfBins, maxBins);
      }
      return 1;
    }
    const unsigned int nn = numberOfBins + grid.Ny() + 3;
    const unsigned int maxnn = GetGridSize(NumberOfClusters, 1);
    if (nn >= maxnn) {
      if (iThread == 0) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_FIRSTHITINBIN_OVERFLOW, iSlice, nn, maxnn);
      }
      return 1;
    }

    calink* bins = &binMemory[RowOffset]; // Reuse mLinkUpData memory as temporary memory

    for (int bin = iThread; bin < numberOfBins; bin += nThreads) {
      c[bin] = 0; // initialize to 0
    }
    GPUbarrier();
    for (int hitIndex = iThread; hitIndex < row.mNHits; hitIndex += nThreads) {
      const int globalHitIndex = RowOffset + hitIndex;
      const calink bin = row.mGrid.GetBin(YZData[globalHitIndex].x, YZData[globalHitIndex].y);

      bins[hitIndex] = bin;
      CAMath::AtomicAdd(&c[bin], 1u);
    }
    GPUbarrier();

    if (iThread == 0) {
      calink n = 0;
      for (int bin = 0; bin < numberOfBins; ++bin) { // TODO: Parallelize
        n += c[bin];
        c[bin] = n;
      }
      for (unsigned int bin = numberOfBins; bin < nn; bin++) {
        c[bin] = n;
      }
    }
    GPUbarrier();

    constexpr float maxVal = (((long int)1 << (sizeof(cahit) < 3 ? sizeof(cahit) * 8 : 24)) - 1); // Stay within float precision in any case!
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

    for (int hitIndex = iThread; hitIndex < row.mNHits; hitIndex += nThreads) {
      const calink bin = bins[hitIndex];
      const calink ind = CAMath::AtomicAdd(&c[bin], (calink)-1) - 1; // generate an index for this hit that is >= c[bin] and < c[bin + 1]
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

    if (iThread == 0) {
      const float maxAbsZ = CAMath::Max(CAMath::Abs(tmpMinMax[2]), CAMath::Abs(tmpMinMax[3]));
      if (maxAbsZ > 300 && !mem->param.par.continuousTracking) {
        mem->errorCodes.raiseError(GPUErrors::ERROR_SLICEDATA_Z_OVERFLOW, iSlice, (unsigned int)maxAbsZ);
        return 1;
      }
    }
  }

  return 0;
}
