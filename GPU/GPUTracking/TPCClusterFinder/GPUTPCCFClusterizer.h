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

/// \file Clusterizer.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CLUSTERIZER_H
#define O2_GPU_CLUSTERIZER_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "CfArray2D.h"
#include "PackedCharge.h"

namespace o2::tpc
{
struct ClusterNative;
} // namespace o2::tpc

namespace o2::gpu
{

class ClusterAccumulator;
class MCLabelAccumulator;

class GPUTPCCFClusterizer : public GPUKernelTemplate
{
 public:
  static constexpr size_t SCRATCH_PAD_WORK_GROUP_SIZE = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFClusterizer);
  struct GPUSharedMemory {
    CfChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_BUILD_N];
    uint8_t innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];
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

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t);

  static GPUd() void computeClustersImpl(int32_t, int32_t, int32_t, int32_t, processorType&, const CfFragment&, GPUSharedMemory&, const CfArray2D<PackedCharge>&, const CfChargePos*, const GPUSettingsRec&, MCLabelAccumulator*, uint32_t, uint32_t, uint32_t*, tpc::ClusterNative*, uint32_t*, int8_t);

  static GPUd() void buildCluster(const GPUSettingsRec&, const CfArray2D<PackedCharge>&, CfChargePos, CfChargePos*, PackedCharge*, uint8_t*, ClusterAccumulator*, MCLabelAccumulator*);

  static GPUd() uint32_t sortIntoBuckets(processorType&, const tpc::ClusterNative&, uint32_t, uint32_t, uint32_t*, tpc::ClusterNative*);

 private:
  static GPUd() void updateClusterInner(const GPUSettingsRec&, uint16_t, uint16_t, const PackedCharge*, const CfChargePos&, ClusterAccumulator*, MCLabelAccumulator*, uint8_t*);

  static GPUd() void updateClusterOuter(uint16_t, uint16_t, uint16_t, uint16_t, const PackedCharge*, const CfChargePos&, ClusterAccumulator*, MCLabelAccumulator*);
};

} // namespace o2::gpu

#endif
