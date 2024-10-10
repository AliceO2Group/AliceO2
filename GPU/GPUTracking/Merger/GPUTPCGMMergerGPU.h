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

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMMergerGeneral : public GPUKernelTemplate
{
 public:
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCMerging; }
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  typedef GPUTPCGMMerger processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return &processors.tpcMerger;
  }
#endif
};

class GPUTPCGMMergerTrackFit : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t mode);
#endif
};

class GPUTPCGMMergerFollowLoopers : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerSliceRefit : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t iSlice);
#endif
};

class GPUTPCGMMergerUnpackGlobal : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t iSlice);
#endif
};

class GPUTPCGMMergerUnpackSaveNumber : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t id);
#endif
};

class GPUTPCGMMergerUnpackResetIds : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t id);
#endif
};

class GPUTPCGMMergerResolve : public GPUTPCGMMergerGeneral
{
 public:
  struct GPUSharedMemory : public gputpcgmmergertypes::GPUResolveSharedMemory {
  };

#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);
#endif
};

class GPUTPCGMMergerClearLinks : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int8_t nOutput);
#endif
};

class GPUTPCGMMergerMergeWithinPrepare : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerMergeSlicesPrepare : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int32_t border0, int32_t border1, int8_t useOrigTrackParam);
#endif
};

class GPUTPCGMMergerMergeBorders : public GPUTPCGMMergerGeneral
{
 public:
  enum K { defaultKernel = 0,
           step0 = 0,
           step1 = 1,
           step2 = 2,
           variant = 3 };
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, Args... args);
#endif
};

class GPUTPCGMMergerMergeCE : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerLinkGlobalTracks : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerCollect : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerPrepareClusters : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerSortTracks : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerSortTracksQPt : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerSortTracksPrepare : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerFinalize : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerMergeLoopers : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
