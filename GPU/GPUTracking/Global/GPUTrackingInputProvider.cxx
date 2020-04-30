// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingInputProvider.cxx
/// \author David Rohr

#include "GPUTrackingInputProvider.h"
#include "GPUDataTypes.h"
#include "GPUReconstruction.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

void GPUTrackingInputProvider::InitializeProcessor() {}
void* GPUTrackingInputProvider::SetPointersInputZS(void* mem)
{
  if (mHoldTPCZS) {
    computePointerWithAlignment(mem, mPzsMeta);
    computePointerWithAlignment(mem, mPzsSizes, GPUTrackingInOutZS::NSLICES * GPUTrackingInOutZS::NENDPOINTS);
    computePointerWithAlignment(mem, mPzsPtrs, GPUTrackingInOutZS::NSLICES * GPUTrackingInOutZS::NENDPOINTS);
  }
  return mem;
}

void* GPUTrackingInputProvider::SetPointersInputClusterNativeAccess(void* mem)
{
  if (mHoldTPCClusterNative) {
    computePointerWithAlignment(mem, mPclusterNativeAccess);
  }
  return mem;
}

void* GPUTrackingInputProvider::SetPointersInputClusterNativeBuffer(void* mem)
{
  if (mHoldTPCClusterNative) {
    computePointerWithAlignment(mem, mPclusterNativeBuffer, mNClusterNative);
  }
  return mem;
}

void* GPUTrackingInputProvider::SetPointersInputClusterNativeOutput(void* mem)
{
  if (mHoldTPCClusterNativeOutput) {
    computePointerWithAlignment(mem, mPclusterNativeOutput, mNClusterNative);
  }
  return mem;
}

void GPUTrackingInputProvider::RegisterMemoryAllocation()
{
  mResourceZS = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputZS, GPUMemoryResource::MEMORY_INPUT, "InputZS");
  mResourceClusterNativeAccess = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputClusterNativeAccess, GPUMemoryResource::MEMORY_INPUT, "ClusterNativeAccess");
  mResourceClusterNativeBuffer = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputClusterNativeBuffer, GPUMemoryResource::MEMORY_INPUT_FLAG | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_EXTERNAL | GPUMemoryResource::MEMORY_CUSTOM, "ClusterNativeBuffer");
  mResourceClusterNativeOutput = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputClusterNativeOutput, GPUMemoryResource::MEMORY_OUTPUT_FLAG | GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_CUSTOM, "ClusterNativeOutput");
}

void GPUTrackingInputProvider::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mHoldTPCZS = io.tpcZS && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding);
  mHoldTPCClusterNative = (io.tpcZS || io.tpcPackedDigits || io.clustersNative) && mRec->IsGPU();
  mHoldTPCClusterNativeOutput = (io.tpcZS || io.tpcPackedDigits);
}
