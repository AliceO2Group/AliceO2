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
#include "ChargePos.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

void GPUTPCClusterFinder::InitializeProcessor() {}

void* GPUTPCClusterFinder::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mPmemory, 1);
  return mem;
}

void* GPUTPCClusterFinder::SetPointersInput(void* mem)
{
  if (mNMaxPages && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
    computePointerWithAlignment(mem, mPzs, mNMaxPages * TPCZSHDR::TPC_ZS_PAGE_SIZE);
  }
  if (mNMaxPages == 0 && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
    computePointerWithAlignment(mem, mPdigits, mNMaxDigits);
  }
  return mem;
}

void* GPUTPCClusterFinder::SetPointersZSOffset(void* mem)
{
  if (mNMaxPages) {
    computePointerWithAlignment(mem, mPzsOffsets, (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding) ? mNMaxPages : GPUTrackingInOutZS::NENDPOINTS);
  }
  return mem;
}

void* GPUTPCClusterFinder::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mPclusterInRow, GPUCA_ROW_COUNT);
  computePointerWithAlignment(mem, mPclusterByRow, GPUCA_ROW_COUNT * mNMaxClusterPerRow);
  return mem;
}

void* GPUTPCClusterFinder::SetPointersScratch(void* mem)
{
  computePointerWithAlignment(mem, mPpositions, mNMaxDigits);
  computePointerWithAlignment(mem, mPpeakPositions, mNMaxPeaks);
  computePointerWithAlignment(mem, mPfilteredPeakPositions, mNMaxClusters);
  computePointerWithAlignment(mem, mPisPeak, mNMaxDigits);
  computePointerWithAlignment(mem, mPchargeMap, TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED);
  computePointerWithAlignment(mem, mPpeakMap, TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED);
  if (not mRec->IsGPU() && mRec->GetDeviceProcessingSettings().runMC) {
    computePointerWithAlignment(mem, mPindexMap, TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED);
    computePointerWithAlignment(mem, mPlabelsByRow, GPUCA_ROW_COUNT * mNMaxClusterPerRow);
    computePointerWithAlignment(mem, mPlabelHeaderOffset, GPUCA_ROW_COUNT);
    computePointerWithAlignment(mem, mPlabelDataOffset, GPUCA_ROW_COUNT);
  }
  computePointerWithAlignment(mem, mPbuf, mBufSize * mNBufs);
  return mem;
}

void GPUTPCClusterFinder::RegisterMemoryAllocation()
{
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersInput, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_GPU, "TPCClustererInput");
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersScratch, GPUMemoryResource::MEMORY_SCRATCH, "TPCClustererScratch", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::ClustererScratch, (unsigned short)(mISlice % mRec->GetDeviceProcessingSettings().nTPCClustererLanes)});
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "TPCClustererMemory");
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT, "TPCClustererOutput");
  mZSOffsetId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersZSOffset, GPUMemoryResource::MEMORY_CUSTOM | GPUMemoryResource::MEMORY_CUSTOM_TRANSFER | GPUMemoryResource::MEMORY_INPUT, "TPCClustererZSOffsets");
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

size_t GPUTPCClusterFinder::getNSteps(size_t items) const
{
  if (items == 0) {
    return 0;
  }
  size_t c = 1;
  size_t capacity = mScanWorkGroupSize;
  while (items > capacity) {
    capacity *= mScanWorkGroupSize;
    c++;
  }
  return c;
}
