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

/// \file GPUTPCCFMCLabelFlattener.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_GPUTPCCF_MCLABEL_FLATTENER_H
#define O2_GPU_GPUTPCCF_MCLABEL_FLATTENER_H

#include "GPUGeneralKernels.h"

#include "clusterFinderDefs.h"
#include "GPUTPCClusterFinder.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE::gpu
{

struct GPUTPCLinearLabels;

class GPUTPCCFMCLabelFlattener : public GPUKernelTemplate
{

 public:
  struct GPUSharedMemory {
  };

  enum K : int32_t {
    setRowOffsets,
    flatten,
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

  static void setGlobalOffsetsAndAllocate(GPUTPCClusterFinder&, GPUTPCLinearLabels&);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
