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
#include "GPUO2DataTypes.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCConvert::InitializeProcessor() {}

void* GPUTPCConvert::SetPointersOutput(void* mem)
{
  if (mRec->GetParam().earlyTpcTransform) {
    computePointerWithAlignment(mem, mClusters, mNClustersTotal);
  }
  return mem;
}

void* GPUTPCConvert::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory, 1);
  return mem;
}

void GPUTPCConvert::RegisterMemoryAllocation()
{
  mMemoryResMemory = mRec->RegisterMemoryAllocation(this, &GPUTPCConvert::SetPointersMemory, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_PERMANENT, "TPCConvertMemory");
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCConvert::SetPointersOutput, GPUMemoryResource::MEMORY_OUTPUT, "TPCConvertOutput");
}

void GPUTPCConvert::SetMaxData(const GPUTrackingInOutPointers& io)
{
  if (io.tpcPackedDigits || io.tpcZS) {
    // TODO: Don't do anything for now, set from the outside, should be fixed
  } else if (io.clustersNative) {
    mNClustersTotal = io.clustersNative->nClustersTotal;
  } else {
    mNClustersTotal = 0;
  }
}
