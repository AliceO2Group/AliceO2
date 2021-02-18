// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{

namespace tpc
{
struct ClusterNative;
}

namespace gpu
{

class ClusterAccumulator;
class MCLabelAccumulator;

class GPUTPCCFClusterizer : public GPUKernelTemplate
{

 public:
  struct GPUSharedMemory {
    ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_BUILD_N];
    uchar innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];
  };

#ifdef HAVE_O2HEADERS
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

  template <int iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  static GPUd() void computeClustersImpl(int, int, int, int, const CfFragment&, GPUSharedMemory&, const Array2D<PackedCharge>&, const ChargePos*, MCLabelAccumulator*, uint, uint, uint*, tpc::ClusterNative*);

 private:
  static GPUd() void updateClusterInner(ushort, ushort, const PackedCharge*, const ChargePos&, ClusterAccumulator*, MCLabelAccumulator*, uchar*);

  static GPUd() void updateClusterOuter(ushort, ushort, ushort, ushort, const PackedCharge*, const ChargePos&, ClusterAccumulator*, MCLabelAccumulator*);

  static GPUd() void buildCluster(const Array2D<PackedCharge>&, ChargePos, ChargePos*, PackedCharge*, uchar*, ClusterAccumulator*, MCLabelAccumulator*);

  static GPUd() uint sortIntoBuckets(const tpc::ClusterNative&, uint, uint, uint*, tpc::ClusterNative*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
