// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int mode);
#endif
};

class GPUTPCGMMergerFollowLoopers : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerUnpack : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int iSlice);
#endif
};

class GPUTPCGMMergerUnpackGlobal : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int iSlice);
#endif
};

class GPUTPCGMMergerUnpackSaveNumber : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int id);
#endif
};

class GPUTPCGMMergerUnpackResetIds : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int id);
#endif
};

class GPUTPCGMMergerResolve : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, char useOrigTrackParam, char mergeAll);
#endif
};

class GPUTPCGMMergerMergeWithin : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerMergeSlices : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger, int border0, int border1, char useOrigTrackParam);
#endif
};

class GPUTPCGMMergerMergeCEInit : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerMergeCE : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerLinkGlobalTracks : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerCollect : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerPrepareClusters : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerSortTracks : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerSortTracksPrepare : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

class GPUTPCGMMergerFinalize : public GPUTPCGMMergerGeneral
{
 public:
#if !defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE)
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
