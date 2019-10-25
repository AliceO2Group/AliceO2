// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUGeneralKernels.cxx
/// \author David Rohr

#include "GPUKernelDebugOutput.h"
#include "GPUReconstruction.h"

#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT

using namespace GPUCA_NAMESPACE::gpu;

void GPUKernelDebugOutput::InitializeProcessor() {}

void* GPUKernelDebugOutput::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mDebugOutMemory, memorySize());
  return mem;
}

void GPUKernelDebugOutput::RegisterMemoryAllocation()
{
  mRec->RegisterMemoryAllocation(this, &GPUKernelDebugOutput::SetPointersMemory, GPUMemoryResource::MEMORY_OUTPUT, "DebugMemory");
}

void GPUKernelDebugOutput::SetMaxData(const GPUTrackingInOutPointers& io) {}

#endif
