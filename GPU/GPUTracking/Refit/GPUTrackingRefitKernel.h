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

/// \file GPUTrackingRefitKernel.h
/// \author David Rohr

#ifndef GPUTRACKINGREFITKERNEL_H
#define GPUTRACKINGREFITKERNEL_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{

class GPUTrackingRefitKernel : public GPUKernelTemplate
{
 public:
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCCompression; }

  enum K : int32_t {
    mode0asGPU = 0,
    mode1asTrackParCov = 1,
  };

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);
};

} // namespace o2::gpu

#endif
