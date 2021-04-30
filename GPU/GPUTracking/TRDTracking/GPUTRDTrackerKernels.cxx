// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackerKernels.cxx
/// \author David Rohr

#include "GPUTRDTrackerKernels.h"
#include "GPUTRDGeometry.h"
#include "GPUConstantMem.h"
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

#ifdef HAVE_O2HEADERS
template <int I>
GPUd() auto& getTracker(GPUTRDTrackerKernels::processorType& processors);
template <>
GPUdi() auto& getTracker<0>(GPUTRDTrackerKernels::processorType& processors)
{
  return processors.trdTrackerGPU;
}
template <>
GPUdi() auto& getTracker<1>(GPUTRDTrackerKernels::processorType& processors)
{
  return processors.trdTrackerO2;
}
#else
template <int I>
GPUdi() GPUTRDTrackerGPU& getTracker(GPUTRDTrackerKernels::processorType& processors)
{
  return processors.trdTrackerGPU;
}
#endif

template <int I>
GPUdii() void GPUTRDTrackerKernels::Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  auto& trdTracker = getTracker<I>(processors);
  GPUCA_OPENMP(parallel for if(!trdTracker.GetRec().GetProcessingSettings().ompKernels) num_threads(trdTracker.GetRec().GetProcessingSettings().ompThreads))
  for (int i = get_global_id(0); i < trdTracker.NTracks(); i += get_global_size(0)) {
    trdTracker.DoTrackingThread(i, get_global_id(0));
  }
}

template GPUd() void GPUTRDTrackerKernels::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors);
#ifdef HAVE_O2HEADERS
template GPUd() void GPUTRDTrackerKernels::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors);
#endif
