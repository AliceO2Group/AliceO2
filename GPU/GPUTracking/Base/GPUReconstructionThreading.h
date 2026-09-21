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

/// \file GPUReconstructionThreading.h
/// \author David Rohr

#if !defined(GPURECONSTRUCTIONTHREADING_H)
#define GPURECONSTRUCTIONTHREADING_H

#if !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"

#include <memory>
#include <oneapi/tbb.h>

namespace o2::gpu
{

struct GPUReconstructionThreading {
  std::unique_ptr<tbb::global_control> control;
  std::unique_ptr<tbb::task_arena> allThreads;
  std::unique_ptr<tbb::task_arena> activeThreads;
  std::unique_ptr<tbb::task_arena> outerThreads;
};

} // namespace o2::gpu

#endif

#define GPUCA_TBB_KERNEL_LOOP_HOST(rec, nBlocks, nThreads, iBlock, iThread, vartype, varname, iEnd, code)        \
  for (vartype varname = (iBlock) * (nThreads) + (iThread); varname < iEnd; varname += (nBlocks) * (nThreads)) { \
    code                                                                                                         \
  }

#ifdef GPUCA_GPUCODE
#define GPUCA_TBB_KERNEL_LOOP GPUCA_TBB_KERNEL_LOOP_HOST
#else
#define GPUCA_TBB_KERNEL_LOOP(rec, nBlocks, nThreads, iBlock, iThread, vartype, varname, iEnd, code)                                                                        \
  if (!rec.GetProcessingSettings().inKernelParallel) {                                                                                                                      \
    rec.mThreading->activeThreads->execute([&] {                                                                                                                            \
      tbb::parallel_for(tbb::blocked_range<vartype>((iBlock) * (nThreads) + (iThread), iEnd, (nBlocks) * (nThreads)), [&](const tbb::blocked_range<vartype>& _r_internal) { \
        for (vartype varname = _r_internal.begin(); varname < _r_internal.end(); varname += (nBlocks) * (nThreads)) {                                                       \
          code                                                                                                                                                              \
        }                                                                                                                                                                   \
      });                                                                                                                                                                   \
    });                                                                                                                                                                     \
  } else {                                                                                                                                                                  \
    GPUCA_TBB_KERNEL_LOOP_HOST(rec, nBlocks, nThreads, iBlock, iThread, vartype, varname, iEnd, code)                                                                       \
  }
#endif

#endif
