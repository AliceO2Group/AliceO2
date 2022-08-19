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
  delete[] mMinMaxCN;
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

void* GPUTPCClusterFinder::SetPointersZS(void* mem)
{
  if (mNMaxPages && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
    computePointerWithAlignment(mem, mPzs, mNMaxPages * TPCZSHDR::TPC_ZS_PAGE_SIZE);
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
  computePointerWithAlignment(mem, mPpadIsNoisy, TPC_PADS_IN_SECTOR);
  computePointerWithAlignment(mem, mPpositions, mNMaxDigitsFragment);
  computePointerWithAlignment(mem, mPpeakPositions, mNMaxPeaks);
  computePointerWithAlignment(mem, mPfilteredPeakPositions, mNMaxClusters);
  if (mRec->GetProcessingSettings().runMC) {
    computePointerWithAlignment(mem, mPclusterPosInRow, mNMaxClusters);
  } else {
    mPclusterPosInRow = nullptr;
  }
  computePointerWithAlignment(mem, mPisPeak, mNMaxDigitsFragment);
  computePointerWithAlignment(mem, mPchargeMap, TPCMapMemoryLayout<decltype(*mPchargeMap)>::items(mRec->GetProcessingSettings().overrideClusterizerFragmentLen));
  computePointerWithAlignment(mem, mPpeakMap, TPCMapMemoryLayout<decltype(*mPpeakMap)>::items(mRec->GetProcessingSettings().overrideClusterizerFragmentLen));
  computePointerWithAlignment(mem, mPbuf, mBufSize * mNBufs);
  computePointerWithAlignment(mem, mPclusterByRow, GPUCA_ROW_COUNT * mNMaxClusterPerRow);

  return mem;
}

void GPUTPCClusterFinder::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersInput, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_STACK, "TPCClustererInput");

  int scratchType = GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK;
  if (mRec->GetProcessingSettings().runMC) {
    scratchType |= GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_GPU;
  }
  mScratchId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersScratch, scratchType, "TPCClustererScratch", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::ClustererScratch, (unsigned short)(mISlice % mRec->GetProcessingSettings().nTPCClustererLanes)});

  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "TPCClustererMemory");
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT | GPUMemoryResource::MEMORY_STACK, "TPCClustererOutput");
  mZSId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersZS, GPUMemoryResource::MEMORY_CUSTOM | GPUMemoryResource::MEMORY_CUSTOM_TRANSFER | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_STACK, "TPCClustererZSData", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::ClustererZS, (unsigned short)(mISlice % mRec->GetProcessingSettings().nTPCClustererLanes)});
  mZSOffsetId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersZSOffset, GPUMemoryResource::MEMORY_CUSTOM | GPUMemoryResource::MEMORY_CUSTOM_TRANSFER | GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_STACK, "TPCClustererZSOffsets");
}

void GPUTPCClusterFinder::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mNMaxPeaks = mRec->MemoryScalers()->NTPCPeaks(mNMaxDigitsFragment);
  mNMaxClusters = mRec->MemoryScalers()->NTPCClusters(mNMaxDigitsFragment, true);
  mNMaxClusterPerRow = 0.01f * mRec->MemoryScalers()->NTPCClusters(mNMaxDigits, true); // TODO: Can save some memory hery by using mNMaxClusters, and copying the computed clusters out after every fragment
  if (io.settingsTF && io.settingsTF->hasNHBFPerTF) {
    unsigned int threshold = 300000 * io.settingsTF->nHBFPerTF / 128;                                                            // TODO: Probably one would need to do this on a row-basis for a better estimate, but currently not supported
    mNMaxClusterPerRow = std::max<unsigned int>(mNMaxClusterPerRow, std::min<unsigned int>(threshold, mNMaxClusterPerRow * 10)); // Relative increased value up until a threshold, for noisy pads
    mNMaxClusterPerRow = std::max<unsigned int>(mNMaxClusterPerRow, io.settingsTF->nHBFPerTF * 20000 / 256);                     // Absolute increased value, to have a minimum for noisy pads
  }
  if (mNMaxDigitsEndpoint) {
    mNMaxClusterPerRow = std::max<unsigned int>(mNMaxClusterPerRow, 0.0085f * mRec->MemoryScalers()->NTPCClusters(mNMaxDigitsEndpoint * GPUTrackingInOutZS::NENDPOINTS, true));
  }
  if (mRec->GetProcessingSettings().tpcIncreasedMinClustersPerRow) {
    mNMaxClusterPerRow = std::max<unsigned int>(mNMaxClusterPerRow, mRec->GetProcessingSettings().tpcIncreasedMinClustersPerRow);
  }

  mBufSize = nextMultipleOf<std::max<int>(GPUCA_MEMALIGN, mScanWorkGroupSize)>(mNMaxDigitsFragment);
  mNBufs = getNSteps(mBufSize);
}

void GPUTPCClusterFinder::SetNMaxDigits(size_t nDigits, size_t nPages, size_t nDigitsFragment, size_t nDigitsEndpointMax)
{
  mNMaxDigits = nextMultipleOf<std::max<int>(GPUCA_MEMALIGN, mScanWorkGroupSize)>(nDigits);
  mNMaxPages = nPages;
  mNMaxDigitsFragment = nDigitsFragment;
  mNMaxDigitsEndpoint = nDigitsEndpointMax;
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
  mPindexMap = new uint[TPCMapMemoryLayout<decltype(*mPindexMap)>::items(mRec->GetProcessingSettings().overrideClusterizerFragmentLen)];
  mPlabelsByRow = new GPUTPCClusterMCInterimArray[GPUCA_ROW_COUNT];
  mPlabelsInRow = new uint[GPUCA_ROW_COUNT];
}

void GPUTPCClusterFinder::clearMCMemory()
{
  delete[] mPindexMap;
  mPindexMap = nullptr;
  delete[] mPlabelsByRow;
  mPlabelsByRow = nullptr;
  delete[] mPlabelsInRow;
  mPlabelsInRow = nullptr;
}
