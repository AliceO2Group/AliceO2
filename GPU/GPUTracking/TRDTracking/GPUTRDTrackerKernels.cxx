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

/// \file GPUTRDTrackerKernels.cxx
/// \author David Rohr

#include "GPUTRDTrackerKernels.h"
#include "GPUTRDGeometry.h"
#include "GPUConstantMem.h"
#include "GPUCommonTypeTraits.h"
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

template <int I, class T>
GPUdii() void GPUTRDTrackerKernels::Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, T* externalInstance)
{
  auto* trdTracker = &processors.getTRDTracker<I>();
#ifndef GPUCA_GPUCODE_DEVICE
#if defined(__cplusplus) && __cplusplus >= 201703L
  if constexpr (std::is_same_v<decltype(trdTracker), decltype(externalInstance)>)
#endif
  {
    if (externalInstance) {
      trdTracker = externalInstance;
    }
  }
#endif
  GPUCA_OPENMP(parallel for if(!trdTracker->GetRec().GetProcessingSettings().ompKernels) num_threads(trdTracker->GetRec().GetProcessingSettings().ompThreads))
  for (int i = get_global_id(0); i < trdTracker->NTracks(); i += get_global_size(0)) {
    trdTracker->DoTrackingThread(i, get_global_id(0));
  }
}

template GPUd() void GPUTRDTrackerKernels::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, GPUTRDTrackerGPU* externalInstance);
#ifdef GPUCA_HAVE_O2HEADERS
template GPUd() void GPUTRDTrackerKernels::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, GPUTRDTracker* externalInstance);
#endif // GPUCA_HAVE_O2HEADERS
