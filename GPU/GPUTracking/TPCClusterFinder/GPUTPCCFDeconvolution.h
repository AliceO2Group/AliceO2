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

/// \file Deconvolution.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_DECONVOLUTION_H
#define O2_GPU_DECONVOLUTION_H

#include "clusterFinderDefs.h"
#include "CfUtils.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE::gpu
{

class GPUTPCCFDeconvolution : public GPUKernelTemplate
{
 public:
  static constexpr size_t SCRATCH_PAD_WORK_GROUP_SIZE = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFDeconvolution);
  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<int16_t, SCRATCH_PAD_WORK_GROUP_SIZE> {
    ChargePos posBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
    uint8_t aboveThresholdBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    uint8_t buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_COUNT_N];
  };

#ifdef GPUCA_HAVE_O2HEADERS
  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }
#endif

  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep()
  {
    return GPUDataTypes::RecoStep::TPCClusterFinding;
  }

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

 private:
  static GPUd() void deconvolutionImpl(int32_t, int32_t, int32_t, int32_t, GPUSharedMemory&, const Array2D<uint8_t>&, Array2D<PackedCharge>&, const ChargePos*, const uint32_t);

  static GPUdi() uint8_t countPeaksInner(uint16_t, const uint8_t*, uint8_t*);
  static GPUdi() uint8_t countPeaksOuter(uint16_t, uint8_t, const uint8_t*);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
