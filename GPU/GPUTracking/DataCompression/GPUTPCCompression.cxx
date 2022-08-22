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

/// \file GPUTPCCompression.cxx
/// \author David Rohr

#include "GPUTPCCompression.h"
#include "GPUReconstruction.h"
#include "GPUO2DataTypes.h"
#include "GPUMemorySizeScalers.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCCompression::InitializeProcessor() {}

void* GPUTPCCompression::SetPointersOutputGPU(void* mem)
{
  SetPointersCompressedClusters(mem, *mOutputA, mOutputA->nAttachedClusters, mOutputA->nTracks, mOutputA->nUnattachedClusters, true);
  return mem;
}

void* GPUTPCCompression::SetPointersOutputHost(void* mem)
{
  computePointerWithoutAlignment(mem, mOutputFlat);
  SetPointersCompressedClusters(mem, *mOutput, mOutput->nAttachedClusters, mOutput->nTracks, mOutput->nUnattachedClusters, true);
  return mem;
}

void* GPUTPCCompression::SetPointersScratch(void* mem)
{
  computePointerWithAlignment(mem, mClusterStatus, mMaxClusters);
  if (mRec->GetProcessingSettings().tpcCompressionGatherMode >= 2) {
    computePointerWithAlignment(mem, mAttachedClusterFirstIndex, mMaxTracks);
  }
  if (mRec->GetProcessingSettings().tpcCompressionGatherMode != 1) {
    SetPointersCompressedClusters(mem, mPtrs, mMaxTrackClusters, mMaxTracks, mMaxClustersInCache, false);
  }
  return mem;
}

void* GPUTPCCompression::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mAttachedClusterFirstIndex, mMaxTrackClusters);
  if (mRec->GetProcessingSettings().tpcCompressionGatherMode == 1) {
    SetPointersCompressedClusters(mem, mPtrs, mMaxTrackClusters, mMaxTracks, mMaxClustersInCache, false);
  }
  return mem;
}

template <class T>
void GPUTPCCompression::SetPointersCompressedClusters(void*& mem, T& c, unsigned int nClA, unsigned int nTr, unsigned int nClU, bool reducedClA)
{
  computePointerWithAlignment(mem, c.qTotU, nClU); // Do not reorder, qTotU ist used as first address in GPUChainTracking::RunTPCCompression
  computePointerWithAlignment(mem, c.qMaxU, nClU);
  computePointerWithAlignment(mem, c.flagsU, nClU);
  computePointerWithAlignment(mem, c.padDiffU, nClU);
  computePointerWithAlignment(mem, c.timeDiffU, nClU);
  computePointerWithAlignment(mem, c.sigmaPadU, nClU);
  computePointerWithAlignment(mem, c.sigmaTimeU, nClU);
  computePointerWithAlignment(mem, c.nSliceRowClusters, GPUCA_ROW_COUNT * NSLICES);

  unsigned int nClAreduced = reducedClA ? nClA - nTr : nClA;

  if (!(mRec->GetParam().rec.tpc.compressionTypeMask & GPUSettings::CompressionTrackModel)) {
    return; // Track model disabled, do not allocate memory
  }
  computePointerWithAlignment(mem, c.qTotA, nClA);
  computePointerWithAlignment(mem, c.qMaxA, nClA);
  computePointerWithAlignment(mem, c.flagsA, nClA);
  computePointerWithAlignment(mem, c.rowDiffA, nClAreduced);
  computePointerWithAlignment(mem, c.sliceLegDiffA, nClAreduced);
  computePointerWithAlignment(mem, c.padResA, nClAreduced);
  computePointerWithAlignment(mem, c.timeResA, nClAreduced);
  computePointerWithAlignment(mem, c.sigmaPadA, nClA);
  computePointerWithAlignment(mem, c.sigmaTimeA, nClA);

  computePointerWithAlignment(mem, c.qPtA, nTr);
  computePointerWithAlignment(mem, c.rowA, nTr);
  computePointerWithAlignment(mem, c.sliceA, nTr);
  computePointerWithAlignment(mem, c.timeA, nTr);
  computePointerWithAlignment(mem, c.padA, nTr);

  computePointerWithAlignment(mem, c.nTrackClusters, nTr);
}

void* GPUTPCCompression::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory);
  computePointerWithAlignment(mem, mOutput);
  mOutputA = mOutput;
  return mem;
}

void GPUTPCCompression::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mMemoryResOutputHost = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersOutputHost, GPUMemoryResource::MEMORY_OUTPUT_FLAG | GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_CUSTOM, "TPCCompressionOutputHost");
  if (mRec->GetProcessingSettings().tpcCompressionGatherMode == 3) {
    mMemoryResOutputGPU = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersOutputGPU, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_CUSTOM | GPUMemoryResource::MEMORY_STACK, "TPCCompressionOutputGPU");
  }
  unsigned int stackScratch = (mRec->GetProcessingSettings().tpcCompressionGatherMode != 3) ? GPUMemoryResource::MEMORY_STACK : 0;
  if (mRec->GetProcessingSettings().tpcCompressionGatherMode < 2) {
    mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT | stackScratch, "TPCCompressionOutput");
  }
  mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersScratch, GPUMemoryResource::MEMORY_SCRATCH | stackScratch, "TPCCompressionScratch");
  mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "TPCCompressionMemory");
}

void GPUTPCCompression::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mMaxClusters = io.clustersNative->nClustersTotal;
  mMaxClusterFactorBase1024 = mMaxClusters > 100000000 ? mRec->MemoryScalers()->NTPCUnattachedHitsBase1024(mRec->GetParam().rec.tpc.rejectionStrategy) : 1024;
  mMaxClustersInCache = mMaxClusters * mMaxClusterFactorBase1024 / 1024;
  mMaxTrackClusters = mRec->GetConstantMem().tpcMerger.NOutputTrackClusters();
  mMaxTracks = mRec->GetConstantMem().tpcMerger.NOutputTracks();
  if (mMaxClusters % 16) {
    mMaxClusters += 16 - (mMaxClusters % 16);
  }
}
