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

/// \file GPUTPCNNClusterizer.h
/// \author Christian Sonnabend

#ifndef O2_GPU_CLUSTERIZER_H
#define O2_GPU_CLUSTERIZER_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"
#include "ML/onnx_interface.h"

using namespace o2::ml;

namespace o2::tpc
{
struct ClusterNative;
} // namespace o2::tpc

namespace GPUCA_NAMESPACE::gpu
{

class ClusterAccumulator;
class MCLabelAccumulator;

class GPUTPCNNClusterizer : public GPUKernelTemplate
{
 public:
  static constexpr size_t SCRATCH_PAD_WORK_GROUP_SIZE = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFClusterizer);
  struct GPUSharedMemory {
    ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_BUILD_N];
    uchar innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];
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

  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, char);

  static GPUd() void computeClustersImpl(int, int, int, int, processorType&, const CfFragment&, GPUSharedMemory&, const Array2D<PackedCharge>&, const ChargePos*, const GPUSettingsRec&, MCLabelAccumulator*, uint, uint, uint*, tpc::ClusterNative*, uint*);

  void exec(int, int, int, int, GPUSharedMemory&, processorType&, char);
  int padOffset(int);
  bool isBoundary(int, int, int);
  static void nn_clusterizer(OnnxModel, OnnxModel,
                              processorType&,
                              const CfFragment&,
                              GPUSharedMemory&,
                              const Array2D<PackedCharge>&,
                              const ChargePos*,
                              const GPUSettingsRec&,
                              MCLabelAccumulator*,
                              uint,
                              uint,
                              uint*,
                              tpc::ClusterNative*,
                              uint*,
                              int = 3, int = 3, int = 3, bool = true);

 private:
  // ---------------------------------
  std::vector<int> pad_row_max{
  65, 65, 65, 67, 67, 67, 69, 69, 69, 71, 71, 71, 73, 73, 73, 73, 75, 75, 75, 75, 77, 77, 77, 79, 79, 79, 81, 81, 81, 83, 83, 83, 85, 85, 85, 87, 87, 87, 89, 89, 89, 89, 91, 91, 91, 93, 93, 93, 91, 91, 91, 93, 93, 93, 95, 95, 95, 97, 97, 97, 99, 99, 99, 75, 75, 75, 75, 77, 77, 77, 79, 79, 79, 79, 81, 81, 81, 83, 83, 83, 83, 85, 85, 85, 87, 87, 87, 89, 89, 89, 89, 91, 91, 91, 93, 93, 93, 93, 95, 95, 95, 97, 97, 97, 99, 99, 101, 101, 101, 103, 103, 103, 105, 109, 109, 111, 111, 111, 113, 113, 113, 115, 115, 115, 117, 117, 117, 117, 117, 119, 119, 121, 121, 123, 123, 123, 125, 125, 127, 127, 127, 129, 129, 131, 131, 131, 133, 133, 135, 135, 137, 137
  };

  static GPUd() void updateClusterInner(const GPUSettingsRec&, ushort, ushort, const PackedCharge*, const ChargePos&, ClusterAccumulator*, MCLabelAccumulator*, uchar*);

  static GPUd() void updateClusterOuter(ushort, ushort, ushort, ushort, const PackedCharge*, const ChargePos&, ClusterAccumulator*, MCLabelAccumulator*);

  static GPUd() void buildCluster(const GPUSettingsRec&, const Array2D<PackedCharge>&, ChargePos, ChargePos*, PackedCharge*, uchar*, ClusterAccumulator*, MCLabelAccumulator*);

  static GPUd() uint sortIntoBuckets(processorType&, const tpc::ClusterNative&, uint, uint, uint*, tpc::ClusterNative*);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
