// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCompression.cxx
/// \author David Rohr

#include "GPUTPCCompression.h"
#include "GPUReconstruction.h"
#include "GPUO2DataTypes.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCCompression::InitializeProcessor() {}

void* GPUTPCCompression::SetPointersOutputHost(void* mem)
{
  SetPointersCompressedClusters(mem, mOutput, mOutput.nAttachedClusters, mOutput.nTracks, mOutput.nUnattachedClusters, true);
  return mem;
}

void* GPUTPCCompression::SetPointersScratch(void* mem)
{
  computePointerWithAlignment(mem, mClusterStatus, mMaxClusters);
  SetPointersCompressedClusters(mem, mPtrs, mMaxTrackClusters, mMaxTracks, mMaxClusters, false);
  computePointerWithAlignment(mem, mClusterSortBuffer, mNGPUBlocks * mNMaxClusterSliceRow);
  return mem;
}

void* GPUTPCCompression::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mAttachedClusterFirstIndex, mMaxTrackClusters);
  return mem;
}

template <class T>
void GPUTPCCompression::SetPointersCompressedClusters(void*& mem, T& c, unsigned int nClA, unsigned int nTr, unsigned int nClU, bool reducedClA)
{
  computePointerWithAlignment(mem, c.qTotU, nClU);
  computePointerWithAlignment(mem, c.qMaxU, nClU);
  computePointerWithAlignment(mem, c.flagsU, nClU);
  computePointerWithAlignment(mem, c.padDiffU, nClU);
  computePointerWithAlignment(mem, c.timeDiffU, nClU);
  computePointerWithAlignment(mem, c.sigmaPadU, nClU);
  computePointerWithAlignment(mem, c.sigmaTimeU, nClU);
  computePointerWithAlignment(mem, c.nSliceRowClusters, GPUCA_ROW_COUNT * NSLICES);

  unsigned int nClAreduced = reducedClA ? nClA - nTr : nClA;

  if (!(mMerger->Param().rec.tpcCompressionModes & GPUSettings::CompressionTrackModel)) {
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
  computePointerWithAlignment(mem, mMemory, 1);
  return mem;
}

void GPUTPCCompression::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mMemoryResOutputHost = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersOutputHost, GPUMemoryResource::MEMORY_OUTPUT_FLAG | GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_CUSTOM, "TPCCompressionOutputHost");
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT, "TPCCompressionOutput");
  mMemoryResScratch = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersScratch, GPUMemoryResource::MEMORY_SCRATCH, "TPCCompressionScratch");
  mMemoryResMemory = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "TPCCompressionMemory");
}

void GPUTPCCompression::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mMaxClusters = mMerger->NMaxClusters();
  mMaxTrackClusters = mMerger->NOutputTrackClusters();
  mMaxTracks = mMerger->NOutputTracks();
  if (mMaxClusters % 16) {
    mMaxClusters += 16 - (mMaxClusters % 16);
  }
}
