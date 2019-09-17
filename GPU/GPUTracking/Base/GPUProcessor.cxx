// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUProcessor.cxx
/// \author David Rohr

#include "GPUProcessor.h"
#include "GPUReconstruction.h"
#include "GPUReconstructionDeviceBase.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUProcessor::GPUProcessor() : mRec(nullptr), mGPUProcessorType(PROCESSOR_TYPE_CPU), mDeviceProcessor(nullptr), mConstantMem(nullptr), mAllocateAndInitializeLate(false) {}

GPUProcessor::~GPUProcessor()
{
  if (mRec && mRec->GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    Clear();
  }
}

void GPUProcessor::InitGPUProcessor(GPUReconstruction* rec, GPUProcessor::ProcessorType type, GPUProcessor* slaveProcessor)
{
  mRec = rec;
  mGPUProcessorType = type;
  if (slaveProcessor) {
    slaveProcessor->mDeviceProcessor = this;
  }
  rec->ConstructGPUProcessor(this);
}

void GPUProcessor::Clear() { mRec->FreeRegisteredMemory(this, true); }
