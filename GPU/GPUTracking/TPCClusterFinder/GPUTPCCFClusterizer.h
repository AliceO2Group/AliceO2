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

class GPUTPCCFClusterizer : public GPUKernelTemplate
{

 public:
  struct GPUSharedMemory {
    ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_BUILD_N];
    uchar innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];
  };

  enum K : int {
    computeClusters,
  };

  static GPUd() void computeClustersImpl(int, int, int, int, GPUSharedMemory&, const Array2D<PackedCharge>&, const deprecated::Digit*, uint, uint, uint*, tpc::ClusterNative*);

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

 private:
  static GPUd() void addOuterCharge(const Array2D<PackedCharge>&, ClusterAccumulator*, const ChargePos&, Delta2);

  static GPUd() Charge addInnerCharge(const Array2D<PackedCharge>&, ClusterAccumulator*, const ChargePos&, Delta2);

  static GPUd() void addCorner(const Array2D<PackedCharge>&, ClusterAccumulator*, const ChargePos&, Delta2);

  static GPUd() void addLine(const Array2D<PackedCharge>&, ClusterAccumulator*, const ChargePos&, Delta2);

  static GPUd() void updateClusterScratchpadInner(ushort, ushort, const PackedCharge*, ClusterAccumulator*, uchar*);

  static GPUd() void updateClusterScratchpadOuter(ushort, ushort, ushort, ushort, const PackedCharge*, ClusterAccumulator*);

  static GPUd() void buildClusterScratchPad(const Array2D<PackedCharge>&, ChargePos, ChargePos*, PackedCharge*, uchar*, ClusterAccumulator*);

  static GPUd() void buildClusterNaive(const Array2D<PackedCharge>&, ClusterAccumulator*, const ChargePos&);

  static GPUd() void sortIntoBuckets(const tpc::ClusterNative&, const uint, const uint, uint*, tpc::ClusterNative*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
