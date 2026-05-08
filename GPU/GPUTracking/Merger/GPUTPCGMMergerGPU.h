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

/// \file GPUTPCGMMergerGPU.h
/// \author David Rohr

#ifndef GPUTPCGMMERGERGPUCA_H
#define GPUTPCGMMERGERGPUCA_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCGMMergerTypes.h"

namespace o2::gpu
{
class GPUTPCGMMergerGeneral : public GPUKernelTemplate
{
 public:
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::TPCMerging; }
  typedef GPUTPCGMMerger processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return &processors.tpcMerger;
  }
};

class GPUTPCGMMergerTrackFit : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t mode);
};

class GPUTPCGMMergerFollowLoopers : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerSectorRefit : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t iSector);
};

class GPUTPCGMMergerUnpackGlobal : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t iSector);
};

class GPUTPCGMMergerUnpackSaveNumber : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t id);
};

class GPUTPCGMMergerUnpackResetIds : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t id);
};

class GPUTPCGMMergerResolve : public GPUTPCGMMergerGeneral
{
 public:
  struct GPUSharedMemory : public gputpcgmmergertypes::GPUResolveSharedMemory {
  };

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);
};

class GPUTPCGMMergerClearLinks : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int8_t nOutput);
};

class GPUTPCGMMergerMergeWithinPrepare : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerMergeSectorsPrepare : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t border0, int32_t border1, int8_t useOrigTrackParam);
};

class GPUTPCGMMergerMergeBorders : public GPUTPCGMMergerGeneral
{
 public:
  enum K { defaultKernel = 0,
           step0 = 0,
           step1 = 1,
           step2 = 2,
           variant = 3 };
  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, Args... args);
};

class GPUTPCGMMergerMergeCE : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerLinkExtrapolatedTracks : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerCollect : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerPrepareForFit : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerSortTracks : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerSortTracksQPt : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerSortTracksPrepare : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerFinalize : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

class GPUTPCGMMergerMergeLoopers : public GPUTPCGMMergerGeneral
{
 public:
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

} // namespace o2::gpu

#endif
