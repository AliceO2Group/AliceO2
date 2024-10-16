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

/// \file GPUTPCDecompression.cxx
/// \author Gabriele Cimador

#include "GPUTPCDecompression.h"
#include "GPUTPCCompression.h"
#include "GPUReconstruction.h"
#include "GPUO2DataTypes.h"
#include "GPUMemorySizeScalers.h"
#include "GPULogging.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCDecompression::InitializeProcessor() {}

void* GPUTPCDecompression::SetPointersInputGPU(void* mem)
{
  SetPointersCompressedClusters(mem, mInputGPU, mInputGPU.nAttachedClusters, mInputGPU.nTracks, mInputGPU.nUnattachedClusters, true);
  return mem;
}

template <class T>
void GPUTPCDecompression::SetPointersCompressedClusters(void*& mem, T& c, uint32_t nClA, uint32_t nTr, uint32_t nClU, bool reducedClA)
{
  computePointerWithAlignment(mem, c.qTotU, nClU); // Do not reorder, qTotU ist used as first address in GPUChainTracking::RunTPCCompression
  computePointerWithAlignment(mem, c.qMaxU, nClU);
  computePointerWithAlignment(mem, c.flagsU, nClU);
  computePointerWithAlignment(mem, c.padDiffU, nClU);
  computePointerWithAlignment(mem, c.timeDiffU, nClU);
  computePointerWithAlignment(mem, c.sigmaPadU, nClU);
  computePointerWithAlignment(mem, c.sigmaTimeU, nClU);
  computePointerWithAlignment(mem, c.nSliceRowClusters, GPUCA_ROW_COUNT * NSLICES);

  uint32_t nClAreduced = reducedClA ? nClA - nTr : nClA;

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

void* GPUTPCDecompression::SetPointersTmpNativeBuffersGPU(void* mem)
{
  computePointerWithAlignment(mem, mTmpNativeClusters, NSLICES * GPUCA_ROW_COUNT * mMaxNativeClustersPerBuffer);
  return mem;
}

void* GPUTPCDecompression::SetPointersTmpNativeBuffersOutput(void* mem)
{
  computePointerWithAlignment(mem, mNativeClustersIndex, NSLICES * GPUCA_ROW_COUNT);
  return mem;
}

void* GPUTPCDecompression::SetPointersTmpNativeBuffersInput(void* mem)
{
  computePointerWithAlignment(mem, mUnattachedClustersOffsets, NSLICES * GPUCA_ROW_COUNT);
  computePointerWithAlignment(mem, mAttachedClustersOffsets, mInputGPU.nTracks);
  return mem;
}

void GPUTPCDecompression::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mMemoryResInputGPU = mRec->RegisterMemoryAllocation(this, &GPUTPCDecompression::SetPointersInputGPU, GPUMemoryResource::MEMORY_INPUT_FLAG | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_EXTERNAL | GPUMemoryResource::MEMORY_SCRATCH, "TPCDecompressionInput");
  mRec->RegisterMemoryAllocation(this, &GPUTPCDecompression::SetPointersTmpNativeBuffersGPU, GPUMemoryResource::MEMORY_SCRATCH, "TPCDecompressionTmpBuffersGPU");
  mResourceTmpIndexes = mRec->RegisterMemoryAllocation(this, &GPUTPCDecompression::SetPointersTmpNativeBuffersOutput, GPUMemoryResource::MEMORY_OUTPUT | GPUMemoryResource::MEMORY_SCRATCH, "TPCDecompressionTmpBuffersOutput");
  mResourceTmpClustersOffsets = mRec->RegisterMemoryAllocation(this, &GPUTPCDecompression::SetPointersTmpNativeBuffersInput, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_SCRATCH, "TPCDecompressionTmpBuffersInput");
}

void GPUTPCDecompression::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mMaxNativeClustersPerBuffer = mRec->GetProcessingSettings().tpcMaxAttachedClustersPerSectorRow;
}
