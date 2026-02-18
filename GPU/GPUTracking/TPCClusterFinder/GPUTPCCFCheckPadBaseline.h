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
///
/// Optionally detects Highly Ionising Particle (HIP) tails: when a saturated
/// ADC value (1023) is found, the tail region on the triggering pad and its
/// neighbors is zeroed in the charge map until charges drop below a
/// configurable threshold.

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

    MaxADC = 1023,

    NThreads = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFCheckPadBaseline),
  };

  union HipTailRange {
    struct {
      int16_t start;
      int16_t end;
    };
    // uint32_t asU32;

    // Be careful with using default initialized values.
    // Need default constructor, so can be placed in shared memory.
    // Might be zero initialized, but invalid tail needs start = end = -1 instead.
    GPUdDefault() HipTailRange() = default;
    GPUdi() HipTailRange(int16_t st, int16_t e) : start(st), end(e) {}

    GPUdi() bool HasValue() const { return start > -1; }
    GPUdi() bool IsOpen() const { return start > -1 && end < 0; }

    GPUdi() void SetOpen(int16_t st)
    {
      start = st;
      end = -1;
    }

    GPUdi() int16_t Length() const { return end - start; }

    GPUdi() void Reset() { start = end = -1; }
  };

  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<int16_t, NThreads> {
    tpccf::Charge charges[NumOfCachedTBs][MaxNPadsPerRow];
    HipTailRange tails[MaxNPadsPerRow];
    uint8_t tailsClosedPad[MaxNPadsPerRow];
    HipTailRange tailsClosed[MaxNPadsPerRow];
  };

  // Accumulated values from scanning cached charges in a pad
  struct PadChargeAccu {
    int32_t totalCharges = 0;
    int32_t consecCharges = 0;
    int32_t maxConsecCharges = 0;
    tpccf::Charge maxCharge = 0;
    int16_t HIPtb = -1;
    HipTailRange activeHIPTail{-1, -1};
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
    const int32_t nBlocks = TPC_PADS_IN_SECTOR / PadsPerCacheline;
    return isGPU ? GPUCA_ROW_COUNT : nBlocks;
  }

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer);

 private:
  GPUd() static void CheckBaselineGPU(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer);
  GPUd() static void CheckBaselineCPU(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer);

  template <int32_t PadsPerBlock>
  GPUd() static CfChargePos padToCfChargePos(int32_t& pad, const GPUTPCClusterFinder&, int32_t& padsPerRow);

  struct RowInfo {
    int16_t globalPadOffset;
    int16_t nPads;
  };
  GPUd() static RowInfo GetRowInfo(int16_t row);

  GPUd() static void updatePadBaseline(int32_t pad, const GPUTPCClusterFinder&, int32_t totalCharges, int32_t consecCharges, tpccf::Charge maxCharge);
};

} // namespace o2::gpu

#endif
