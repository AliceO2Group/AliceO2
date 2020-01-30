// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterFinderKernels.h
/// \author David Rohr

#ifndef O2_GPU_GPUTPCCLUSTERFINDERKERNEL_H
#define O2_GPU_GPUTPCCLUSTERFINDERKERNEL_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCSharedMemoryData.h"

#include "clusterFinderDefs.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCClusterFinderKernels : public GPUKernelTemplate
{
 public:
  class GPUTPCSharedMemory : public GPUKernelTemplate::GPUTPCSharedMemoryScan64<int, GPUCA_THREAD_COUNT_SCAN>
  {
   public:
    union {
      GPUTPCSharedMemoryData::search_t search;
      GPUTPCSharedMemoryData::noise_t noise;
      GPUTPCSharedMemoryData::count_t count;
      GPUTPCSharedMemoryData::build_t build;
      GPUTPCSharedMemoryData::zs_t zs;
    };
  };

  enum K : int {
    fillChargeMap = 0,
    resetMaps = 1,
    findPeaks = 2,
    noiseSuppression = 3,
    updatePeaks = 4,
    countPeaks = 5,
    computeClusters = 6,
    nativeScanUpStart = 7,
    nativeScanUp = 8,
    nativeScanTop = 9,
    nativeScanDown = 10,
    compactDigit = 11,
    decodeZS = 12
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
  template <int iKernel = 0, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, Args... args);

 private:
  GPUd() static int compactionElems(processorType& clusterer, int stage);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
