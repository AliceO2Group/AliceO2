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
  if (mHoldTPCZS && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
    computePointerWithAlignment(mem, mPzsMeta);
    computePointerWithAlignment(mem, mPzsSizes, GPUTrackingInOutZS::NSLICES * GPUTrackingInOutZS::NENDPOINTS);
    computePointerWithAlignment(mem, mPzsPtrs, GPUTrackingInOutZS::NSLICES * GPUTrackingInOutZS::NENDPOINTS);
  }
  return mem;
}

void* GPUTrackingInputProvider::SetPointersInputClusterNativeAccess(void* mem)
{
  if (mHoldTPCClusterNative && mRec->IsGPU()) {
    computePointerWithAlignment(mem, mPclusterNativeAccess);
  }
  return mem;
}

void* GPUTrackingInputProvider::SetPointersInputClusterNativeBuffer(void* mem)
{
  if (mHoldTPCClusterNative && mRec->IsGPU()) {
    computePointerWithAlignment(mem, mPclusterNativeBuffer, mNClusterNative);
  }
  return mem;
}

void GPUTrackingInputProvider::RegisterMemoryAllocation()
{
  mResourceZS = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputZS, GPUMemoryResource::MEMORY_INPUT, "InputZS");
  mResourceClusterNativeAccess = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputClusterNativeAccess, GPUMemoryResource::MEMORY_INPUT, "ClusterNativeAccess");
  mResourceClusterNativeBuffer = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputClusterNativeBuffer, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_EXTERNAL | GPUMemoryResource::MEMORY_CUSTOM_TRANSFER | GPUMemoryResource::MEMORY_CUSTOM, "ClusterNativeBuffer");
}

void GPUTrackingInputProvider::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mHoldTPCZS = io.tpcZS;
  mHoldTPCClusterNative = io.tpcZS || io.tpcPackedDigits || io.clustersNative;
}
