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

/// \file GPUProcessor.cxx
/// \author David Rohr

#include "GPUProcessor.h"
#include "GPUReconstruction.h"
#include "GPUReconstructionDeviceBase.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUProcessor::GPUProcessor() : mRec(nullptr), mGPUProcessorType(PROCESSOR_TYPE_CPU), mLinkedProcessor(nullptr), mConstantMem(nullptr), mAllocateAndInitializeLate(false) {}

GPUProcessor::~GPUProcessor()
{
  if (mRec && mRec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    Clear();
  }
}

void GPUProcessor::InitGPUProcessor(GPUReconstruction* rec, GPUProcessor::ProcessorType type, GPUProcessor* slaveProcessor)
{
  mRec = rec;
  mGPUProcessorType = type;
  if (slaveProcessor) {
    slaveProcessor->mLinkedProcessor = this;
    mLinkedProcessor = slaveProcessor;
  }
  rec->ConstructGPUProcessor(this);
}

void GPUProcessor::Clear() { mRec->FreeRegisteredMemory(this, true); }
