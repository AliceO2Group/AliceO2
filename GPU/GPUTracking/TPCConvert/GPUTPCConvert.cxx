// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCConvert.cxx
/// \author David Rohr

#include "GPUTPCConvert.h"
#include "TPCFastTransform.h"
#include "GPUTPCClusterData.h"
#include "GPUReconstruction.h"
#include "ClusterNativeAccessExt.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCConvert::InitializeProcessor() {}

void* GPUTPCConvert::SetPointersInput(void* mem)
{
  computePointerWithAlignment(mem, mInputClusters, mNClustersTotal);
  return mem;
}

void* GPUTPCConvert::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mClusters, mNClustersTotal);
  return mem;
}

void* GPUTPCConvert::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory, 1);
  if (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion) {
    computePointerWithAlignment(mem, mClustersNativeBuffer, 1);
  }
  return mem;
}

void GPUTPCConvert::RegisterMemoryAllocation()
{
  mMemoryResMemory = mRec->RegisterMemoryAllocation(this, &GPUTPCConvert::SetPointersMemory, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_PERMANENT, "TPCConvertMemory");
  mMemoryResInput = mRec->RegisterMemoryAllocation(this, &GPUTPCConvert::SetPointersInput, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_EXTERNAL | GPUMemoryResource::MEMORY_CUSTOM_TRANSFER, "TPCConvertInput");
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCConvert::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT, "TPCConvertOutput");
}

void GPUTPCConvert::SetMaxData()
{
  unsigned int offset = 0;
  if (mClustersNative) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < o2::TPC::Constants::MAXGLOBALPADROW; j++) {
        mClustersNative->clusterOffset[i][j] = offset;
        offset += mClustersNative->nClusters[i][j];
      }
    }
  }
  mNClustersTotal = offset;
}

void GPUTPCConvert::set(ClusterNativeAccessExt* clustersNative, const TPCFastTransform* transform)
{
  mClustersNative = clustersNative;
  mTransform = transform;
}
