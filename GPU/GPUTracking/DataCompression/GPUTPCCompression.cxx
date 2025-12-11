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
#include "GPUDefParametersRuntime.h"
#include "GPUConstantMem.h"

using namespace o2::gpu;

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
  int32_t gatherMode = mRec->GetProcessingSettings().tpcCompressionGatherMode == -1 ? mRec->getGPUParameters(mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression).par_COMP_GATHER_MODE : mRec->GetProcessingSettings().tpcCompressionGatherMode;
  computePointerWithAlignment(mem, mClusterStatus, mMaxClusters);
  if (gatherMode >= 2) {
    computePointerWithAlignment(mem, mAttachedClusterFirstIndex, mMaxTracks);
  }
  if (gatherMode != 1) {
    SetPointersCompressedClusters(mem, mPtrs, mMaxTrackClusters, mMaxTracks, mMaxClustersInCache, false);
  }
  return mem;
}

void* GPUTPCCompression::SetPointersOutput(void* mem)
{
  int32_t gatherMode = mRec->GetProcessingSettings().tpcCompressionGatherMode == -1 ? mRec->getGPUParameters(mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression).par_COMP_GATHER_MODE : mRec->GetProcessingSettings().tpcCompressionGatherMode;
  computePointerWithAlignment(mem, mAttachedClusterFirstIndex, mMaxTrackClusters);
  if (gatherMode == 1) {
    SetPointersCompressedClusters(mem, mPtrs, mMaxTrackClusters, mMaxTracks, mMaxClustersInCache, false);
  }
  return mem;
}

template <class T>
void GPUTPCCompression::SetPointersCompressedClusters(void*& mem, T& c, uint32_t nClA, uint32_t nTr, uint32_t nClU, bool reducedClA)
{
  computePointerWithAlignment(mem, c.qTotU, nClU); // Do not reorder, qTotU ist used as first address in GPUChainTracking::RunTPCCompression
  computePointerWithAlignment(mem, c.qMaxU, nClU);
  computePointerWithAlignment(mem, c.flagsU, nClU);
  computePointerWithAlignment(mem, c.padDiffU, nClU);
  computePointerWithAlignment(mem, c.timeDiffU, nClU);
  computePointerWithAlignment(mem, c.sigmaPadU, nClU);
  computePointerWithAlignment(mem, c.sigmaTimeU, nClU);
  computePointerWithAlignment(mem, c.nSliceRowClusters, GPUCA_ROW_COUNT * NSECTORS);

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
  int32_t gatherMode = mRec->GetProcessingSettings().tpcCompressionGatherMode == -1 ? mRec->getGPUParameters(mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression).par_COMP_GATHER_MODE : mRec->GetProcessingSettings().tpcCompressionGatherMode;
  mMemoryResOutputHost = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersOutputHost, GPUMemoryResource::MEMORY_OUTPUT_FLAG | GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_CUSTOM, "TPCCompressionOutputHost");
  if (gatherMode == 3) {
    mMemoryResOutputGPU = mRec->RegisterMemoryAllocation(this, &GPUTPCCompression::SetPointersOutputGPU, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_CUSTOM | GPUMemoryResource::MEMORY_STACK, "TPCCompressionOutputGPU");
  }
  uint32_t stackScratch = (gatherMode != 3) ? GPUMemoryResource::MEMORY_STACK : 0; // TODO: Can we use stacked memory also with gather mode 3?
  if (gatherMode < 2) {
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
  mMaxTrackClusters = mRec->GetConstantMem().tpcMerger.NMergedTrackClusters(); // TODO: Why is this not using ioPtrs? Could remove GPUConstantMem.h include
  mMaxTracks = mRec->GetConstantMem().tpcMerger.NMergedTracks();
  if (mMaxClusters % 16) {
    mMaxClusters += 16 - (mMaxClusters % 16);
  }
}

void GPUTPCCompression::DumpCompressedClusters(std::ostream& out)
{
  const o2::tpc::CompressedClusters O = *mOutputFlat;
  out << "\n\nCompressed Clusters:\n";
  out << O.nTracks << " Tracks\n";
  out << "Slice Row Clusters:\n";
  for (uint32_t i = 0; i < NSECTORS; i++) {
    out << "Sector " << i << ": ";
    for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
      out << (O.nSliceRowClusters ? O.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] : 0) << ", ";
    }
    out << "\n";
  }
  out << "\nTrack Clusters:\n";
  for (uint32_t i = 0; i < O.nTracks; i++) {
    if (i && i % 100 == 0) {
      out << "\n";
    }
    out << O.nTrackClusters[i] << ", ";
  }
  out << "\n\nUnattached Clusters\n";
  uint32_t offset = 0;
  if (O.nSliceRowClusters) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
        out << "Sector " << i << " Row " << j << ": ";
        for (uint32_t k = 0; k < O.nSliceRowClusters[i * GPUCA_ROW_COUNT + j]; k++) {
          if (k && k % 10 == 0) {
            out << "\n    ";
          }
          const uint32_t l = k + offset;
          out << "[" << (uint32_t)O.qTotU[l] << ", " << (uint32_t)O.qMaxU[l] << ", " << (uint32_t)O.flagsU[l] << ", " << (int32_t)O.padDiffU[l] << ", " << (int32_t)O.timeDiffU[l] << ", " << (uint32_t)O.sigmaPadU[l] << ", " << (uint32_t)O.sigmaTimeU[l] << "] ";
        }
        offset += O.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
        out << "\n";
      }
    }
  }
  out << "\n\nAttached Clusters\n";
  offset = 0;
  for (uint32_t i = 0; i < O.nTracks; i++) {
    out << "Track " << i << ": {" << (uint32_t)O.qPtA[i] << ", " << (uint32_t)O.rowA[i] << ", " << (uint32_t)O.sliceA[i] << ", " << (uint32_t)O.timeA[i] << ", " << (uint32_t)O.padA[i] << "} - ";
    for (uint32_t k = 0; k < O.nTrackClusters[i]; k++) {
      if (k && k % 10 == 0) {
        out << "\n    ";
      }
      const uint32_t l1 = offset + k, l2 = offset - i + k - 1;
      out << "[";
      if (k) {
        out << (int32_t)O.rowDiffA[l2] << ", " << (int32_t)O.sliceLegDiffA[l2] << ", " << (uint32_t)O.padResA[l2] << ", " << (uint32_t)O.timeResA[l2] << ", ";
      }
      out << (uint32_t)O.qTotA[l1] << ", " << (uint32_t)O.qMaxA[l1] << ", " << (uint32_t)O.flagsA[l1] << ", " << (uint32_t)O.sigmaPadA[l1] << ", " << (uint32_t)O.sigmaTimeA[l1] << "] ";
    }
    offset += O.nTrackClusters[i];
    out << "\n";
  }
}
