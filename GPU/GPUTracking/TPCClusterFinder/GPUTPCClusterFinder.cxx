// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterFinder.cxx
/// \author David Rohr

#include "GPUTPCClusterFinder.h"
#include "GPUReconstruction.h"
#include "GPUMemorySizeScalers.h"
#include "GPUHostDataTypes.h"

#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/Digit.h"

#include "ChargePos.h"
#include "Array2D.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

void GPUTPCClusterFinder::InitializeProcessor()
{
  mMinMaxCN = new MinMaxCN[GPUTrackingInOutZS::NENDPOINTS];
}

GPUTPCClusterFinder::~GPUTPCClusterFinder()
{
  clearMCMemory();
}

void* GPUTPCClusterFinder::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mPmemory, 1);
  return mem;
}

void* GPUTPCClusterFinder::SetPointersInput(void* mem)
{
  if (mNMaxPages == 0 && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
    computePointerWithAlignment(mem, mPdigits, mNMaxDigits);
  }
  return mem;
}

void* GPUTPCClusterFinder::SetPointersZSOffset(void* mem)
{
  const int n = (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding) ? mNMaxPages : GPUTrackingInOutZS::NENDPOINTS;
  if (n) {
    computePointerWithAlignment(mem, mPzsOffsets, n);
  }
  return mem;
}

void* GPUTPCClusterFinder::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mPclusterInRow, GPUCA_ROW_COUNT);
  return mem;
}

void* GPUTPCClusterFinder::SetPointersScratch(void* mem)
{
  if (mNMaxPages && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
    computePointerWithAlignment(mem, mPzs, mNMaxPages * TPCZSHDR::TPC_ZS_PAGE_SIZE);
  }
  computePointerWithAlignment(mem, mPpositions, mNMaxDigits);
  computePointerWithAlignment(mem, mPpeakPositions, mNMaxPeaks);
  computePointerWithAlignment(mem, mPfilteredPeakPositions, mNMaxClusters);
  computePointerWithAlignment(mem, mPisPeak, mNMaxDigits);
  computePointerWithAlignment(mem, mPchargeMap, TPCMapMemoryLayout<decltype(*mPchargeMap)>::items());
  computePointerWithAlignment(mem, mPpeakMap, TPCMapMemoryLayout<decltype(*mPpeakMap)>::items());
  computePointerWithAlignment(mem, mPbuf, mBufSize * mNBufs);
  computePointerWithAlignment(mem, mPclusterByRow, GPUCA_ROW_COUNT * mNMaxClusterPerRow);
  return mem;
}

void GPUTPCClusterFinder::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersInput, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_STACK, "TPCClustererInput");
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersScratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCClustererScratch", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::ClustererScratch, (unsigned short)(mISlice % mRec->GetDeviceProcessingSettings().nTPCClustererLanes)});
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "TPCClustererMemory");
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT | GPUMemoryResource::MEMORY_STACK, "TPCClustererOutput");
  mZSOffsetId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersZSOffset, GPUMemoryResource::MEMORY_CUSTOM | GPUMemoryResource::MEMORY_CUSTOM_TRANSFER | GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_STACK, "TPCClustererZSOffsets");
}

void GPUTPCClusterFinder::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mNMaxPeaks = mRec->MemoryScalers()->NTPCPeaks(mNMaxDigits);
  mNMaxClusters = mRec->MemoryScalers()->NTPCClusters(mNMaxDigits);
  mNMaxClusterPerRow = 0.01f * mNMaxClusters;
  mBufSize = nextMultipleOf<std::max<int>(GPUCA_MEMALIGN, mScanWorkGroupSize)>(mNMaxDigits);
  mNBufs = getNSteps(mBufSize);
}

void GPUTPCClusterFinder::SetNMaxDigits(size_t nDigits, size_t nPages)
{
  mNMaxDigits = nextMultipleOf<std::max<int>(GPUCA_MEMALIGN, mScanWorkGroupSize)>(nDigits);
  mNMaxPages = nPages;
}

unsigned int GPUTPCClusterFinder::getNSteps(size_t items) const
{
  if (items == 0) {
    return 0;
  }
  unsigned int c = 1;
  size_t capacity = mScanWorkGroupSize;
  while (items > capacity) {
    capacity *= mScanWorkGroupSize;
    c++;
  }
  return c;
}

void GPUTPCClusterFinder::PrepareMC()
{
  assert(mNMaxClusterPerRow > 0);

  clearMCMemory();
  mPindexMap = new uint[TPCMapMemoryLayout<decltype(*mPindexMap)>::items()];
  mPlabelsByRow = new GPUTPCClusterMCInterim[GPUCA_ROW_COUNT * mNMaxClusterPerRow];
  mPlabelHeaderOffset = new uint[GPUCA_ROW_COUNT];
  mPlabelDataOffset = new uint[GPUCA_ROW_COUNT];
}

void GPUTPCClusterFinder::clearMCMemory()
{
  delete[] mPindexMap;
  mPindexMap = nullptr;
  delete[] mPlabelsByRow;
  mPlabelsByRow = nullptr;
  delete[] mPlabelHeaderOffset;
  mPlabelHeaderOffset = nullptr;
  delete[] mPlabelDataOffset;
  mPlabelDataOffset = nullptr;
  delete[] mMinMaxCN;
  mMinMaxCN = nullptr;
}
