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

/// \file GPUTPCGlobalDebugSortKernels.h
/// \author David Rohr

#ifndef GPUTPCGLOBALDEBUGSORTKERNELS_H
#define GPUTPCGLOBALDEBUGSORTKERNELS_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE::gpu
{

class GPUTPCGMMerger;
class GPUTPCGlobalDebugSortKernels : public GPUKernelTemplate
{
 public:
  enum K { defaultKernel = 0,
           clearIds = 0,
           sectorTracks = 1,
           globalTracks1 = 2,
           globalTracks2 = 3,
           borderTracks = 4 };
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCMerging; }
  typedef GPUTPCGMMerger processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors) { return &processors.tpcMerger; }

  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker, char parameter);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif // GPUTPCGLOBALDEBUGSORTKERNELS_H
