// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGlobalTracking.h
/// \author David Rohr

#ifndef GPUTPCGLOBALTRACKING_H
#define GPUTPCGLOBALTRACKING_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCTracker;

class GPUTPCGlobalTracking : public GPUKernelTemplate
{
 public:
  struct GPUSharedMemory {
    CA_SHARED_STORAGE(MEM_LG(GPUTPCRow) mRows[GPUCA_ROW_COUNT]);
  };

  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() CONSTEXPRRET static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  GPUhdi() static processorType* Processor(MEM_TYPE(GPUConstantMem) & processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& tracker);

  GPUd() static int GlobalTrackingSliceOrder(int iSlice);

 private:
  GPUd() static int PerformGlobalTrackingRun(GPUTPCTracker& tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & GPUrestrict() smem, const GPUTPCTracker& sliceSource, int iTrack, int rowIndex, float angle, int direction);
  GPUd() static void PerformGlobalTracking(GPUTPCTracker& tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & GPUrestrict() smem);
  GPUd() static void PerformGlobalTracking(const GPUTPCTracker& tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & GPUrestrict() smem, GPUTPCTracker& sliceTarget, bool right);
};

class GPUTPCGlobalTrackingCopyNumbers : public GPUKernelTemplate
{
 public:
  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() CONSTEXPRRET static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& tracker, int n);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKLETCONSTRUCTOR_H
