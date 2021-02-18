// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCompressionKernels.h
/// \author David Rohr

#ifndef GPUTPCCONMPRESSIONKERNELS_H
#define GPUTPCCONMPRESSIONKERNELS_H

#include "GPUGeneralKernels.h"

namespace o2
{
namespace tpc
{
struct ClusterNative;
}
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCCompressionKernels : public GPUKernelTemplate
{
 public:
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCCompression; }

  enum K : int {
    step0attached = 0,
    step1unattached = 1
  };

  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<int, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCompressionKernels_step1unattached)> {
    GPUAtomic(unsigned int) nCount;
    unsigned int lastIndex;
    unsigned int sortBuffer[GPUCA_TPC_COMP_CHUNK_SIZE];
  };

  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors);

 public:
  template <int I>
  class GPUTPCCompressionKernels_Compare
  {
   public:
    GPUhdi() GPUTPCCompressionKernels_Compare(const o2::tpc::ClusterNative* p) : mClsPtr(p) {}
    GPUd() bool operator()(unsigned int a, unsigned int b) const;

   protected:
    const o2::tpc::ClusterNative* mClsPtr;
  };
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
