// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  enum K : int {
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

  template <int iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  static void setGlobalOffsetsAndAllocate(GPUTPCClusterFinder&, GPUTPCLinearLabels&);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
