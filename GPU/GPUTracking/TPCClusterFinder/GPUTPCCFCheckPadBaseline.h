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

/// \file GPUTPCCFCheckPadBaseline.h
/// \author Felix Weiglhofer
///
/// Kernel identifies noisy TPC pads by analyzing charge patterns over time.
/// A pad is marked noisy if it exceeds thresholds for total or consecutive
/// time bins with charge, unless the charge exceeds a saturation threshold.

#ifndef O2_GPU_GPU_TPC_CF_CHECK_PAD_BASELINE_H
#define O2_GPU_GPU_TPC_CF_CHECK_PAD_BASELINE_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCGeometry.h"

#include "clusterFinderDefs.h"
#include "CfArray2D.h"

namespace o2::gpu
{

class GPUTPCCFCheckPadBaseline : public GPUKernelTemplate
{

 public:
  enum {
    PadsPerCacheline = TPCMapMemoryLayout<uint16_t>::Width,
    TimebinsPerCacheline = TPCMapMemoryLayout<uint16_t>::Height,
    EntriesPerCacheline = PadsPerCacheline * TimebinsPerCacheline,
    NumOfCachedPads = GPUCA_WARP_SIZE / TimebinsPerCacheline,
    NumCLsPerWarp = GPUCA_WARP_SIZE / EntriesPerCacheline,
    NumOfCachedTBs = TimebinsPerCacheline,
    // Threads index shared memory as [iThread / MaxNPadsPerRow][iThread % MaxNPadsPerRow].
    // Rounding up to a multiple of PadsPerCacheline ensures iThread / MaxNPadsPerRow < NumOfCachedTBs
    // for all threads, avoiding out-of-bounds access.
    MaxNPadsPerRow = CAMath::nextMultipleOf<PadsPerCacheline>(GPUTPCGeometry::MaxNPadsPerRow()),
  };

  struct GPUSharedMemory {
    tpccf::Charge charges[NumOfCachedTBs][MaxNPadsPerRow];
  };

  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }

  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep()
  {
    return gpudatatypes::RecoStep::TPCClusterFinding;
  }

  static int32_t GetNBlocks(bool isGPU)
  {
    // Important to exclude rightmost padding from Pad Filter.
    // There's nothing to filter there and padding is counted as start of a row, so it causes an overflow in the row count.
    const int32_t nBlocksCPU = (TPC_CLUSTERER_STRIDED_PAD_COUNT - GPUCF_PADDING_PAD) / PadsPerCacheline;
    return isGPU ? GPUTPCGeometry::NROWS : nBlocksCPU;
  }

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer);

 private:
  GPUd() static void CheckBaselineGPU(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer);
  GPUd() static void CheckBaselineCPU(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer);

  GPUd() static void updatePadBaseline(int32_t pad, const GPUTPCClusterFinder&, int32_t totalCharges, int32_t consecCharges, tpccf::Charge maxCharge);
};

} // namespace o2::gpu

#endif
