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
  if (holdsTPCZS && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
    computePointerWithAlignment(mem, mPzsMeta, 1);
    computePointerWithAlignment(mem, mPzsSizes, GPUTrackingInOutZS::NSLICES * GPUTrackingInOutZS::NENDPOINTS);
    computePointerWithAlignment(mem, mPzsPtrs, GPUTrackingInOutZS::NSLICES * GPUTrackingInOutZS::NENDPOINTS);
  }
  return mem;
}

void* GPUTrackingInputProvider::SetPointersInputGPUOnly(void* mem)
{
  return mem;
}

void GPUTrackingInputProvider::RegisterMemoryAllocation()
{
  mResourceZS = mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputZS, GPUMemoryResource::MEMORY_INPUT, "InputZS");
  mRec->RegisterMemoryAllocation(this, &GPUTrackingInputProvider::SetPointersInputGPUOnly, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_GPU, "InputGPU"); // TODO: move more here (example)
}

void GPUTrackingInputProvider::SetMaxData(const GPUTrackingInOutPointers& io)
{
  holdsTPCZS = io.tpcZS;
}
