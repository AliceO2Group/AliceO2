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

/// \file GPUTPCSectorDebugSortKernels.h
/// \author David Rohr

#ifndef GPUTPCSECTORDEBUGSORTKERNELS_H
#define GPUTPCSECTORDEBUGSORTKERNELS_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE::gpu
{
class GPUTPCTracker;

class GPUTPCSectorDebugSortKernels : public GPUKernelTemplate
{
 public:
  enum K { defaultKernel = 0,
           hitData = 0,
           startHits = 1 };
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCSliceTracking; }
  typedef GPUTPCTracker processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors) { return processors.tpcTrackers; }

  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif // GPUTPCSECTORDEBUGSORTKERNELS_H
