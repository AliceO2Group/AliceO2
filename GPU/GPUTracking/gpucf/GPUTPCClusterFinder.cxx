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

namespace gpucf
{
#include "cl/shared/ClusterNative.h"
#include "cl/shared/Digit.h"
} // namespace gpucf

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCClusterFinder::InitializeProcessor() {}

void* GPUTPCClusterFinder::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mPmemory, 1);
  return mem;
}

void* GPUTPCClusterFinder::SetPointersInput(void* mem)
{
  if (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding) {
    computePointerWithAlignment(mem, mPdigits, mNMaxDigits);
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
  computePointerWithAlignment(mem, mPpeaks, mNMaxDigits);
  computePointerWithAlignment(mem, mPfilteredPeaks, mNMaxDigits);
  computePointerWithAlignment(mem, mPisPeak, mNMaxDigits);
  computePointerWithAlignment(mem, mPchargeMap, TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED);
  computePointerWithAlignment(mem, mPpeakMap, TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED);
  computePointerWithAlignment(mem, mPbuf, mBufSize * mNBufs);
  return mem;
}

void GPUTPCClusterFinder::RegisterMemoryAllocation()
{
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersInput, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_GPU, "TPCClustererInput");
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersScratch, GPUMemoryResource::MEMORY_SCRATCH, "TPCClustererScratch", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::ClustererScratch, mISlice % 8}); // TODO: Refine constant 8
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "TPCClustererMemory");
  mRec->RegisterMemoryAllocation(this, &GPUTPCClusterFinder::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT, "TPCClustererOutput");
}

void GPUTPCClusterFinder::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mNMaxPeaks = mNMaxDigits;
  mNMaxClusters = 0.5f * mNMaxPeaks;
  mNMaxClusterPerRow = 0.01f * mNMaxDigits;
  mBufSize = nextMultipleOf<std::max<int>(GPUCA_MEMALIGN_SMALL, mScanWorkGroupSize)>(mNMaxDigits);
  mNBufs = getNSteps(mBufSize);
}

void GPUTPCClusterFinder::SetNMaxDigits(size_t n)
{
  mNMaxDigits = nextMultipleOf<std::max<int>(GPUCA_MEMALIGN_SMALL, mScanWorkGroupSize)>(n);
}

size_t GPUTPCClusterFinder::getNSteps(size_t items) const
{
  size_t c = 0;
  while (items > 0) {
    items /= mScanWorkGroupSize;
    c++;
  }
  return c;
}
