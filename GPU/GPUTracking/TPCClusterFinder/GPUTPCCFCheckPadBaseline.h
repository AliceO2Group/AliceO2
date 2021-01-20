// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCFCheckPadBaseline.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_GPU_TPC_CF_CHECK_PAD_BASELINE_H
#define O2_GPU_GPU_TPC_CF_CHECK_PAD_BASELINE_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

#include "clusterFinderDefs.h"

namespace GPUCA_NAMESPACE::gpu
{

class GPUTPCCFCheckPadBaseline : public GPUKernelTemplate
{

 public:
  enum {
    PadsPerCacheline = 8,
    TimebinsPerCacheline = 4,
    NumOfCachedTimebins = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFCheckPadBaseline) / PadsPerCacheline,
  };

  struct GPUSharedMemory {
    tpccf::Charge charges[PadsPerCacheline][NumOfCachedTimebins];
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

  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer);

 private:
  GPUd() static ChargePos padToChargePos(int& pad, const GPUTPCClusterFinder&);
  GPUd() static void updatePadBaseline(int pad, const GPUTPCClusterFinder&, int totalCharges, int consecCharges);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
