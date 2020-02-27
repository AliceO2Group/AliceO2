// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMMergerGPU.cxx
/// \author David Rohr

#include "GPUTPCGMMergerGPU.h"
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCGMMergerTrackFit::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger)
{
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#pragma omp parallel for num_threads(merger.GetRec().GetDeviceProcessingSettings().nThreads)
#endif
  for (int i = get_global_id(0); i < merger.NOutputTracks(); i += get_global_size(0)) {
    GPUTPCGMTrackParam::RefitTrack(merger.OutputTracks()[i], i, &merger, merger.Clusters());
  }
}
