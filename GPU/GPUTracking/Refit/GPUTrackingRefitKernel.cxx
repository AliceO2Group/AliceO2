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

/// \file GPUTrackingRefitKernel.cxx
/// \author David Rohr

#include "GPUTrackingRefitKernel.h"
#include "GPUTrackingRefit.h"
#include "GPUROOTDump.h"

using namespace GPUCA_NAMESPACE::gpu;

template <int32_t I>
GPUdii() void GPUTrackingRefitKernel::Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  auto& refit = processors.trackingRefit;
  for (uint32_t i = get_global_id(0); i < processors.ioPtrs.nMergedTracks; i += get_global_size(0)) {
    if (refit.mPTracks[i].OK()) {
      GPUTPCGMMergedTrack trk = refit.mPTracks[i];
      int32_t retval;
      if constexpr (I == mode0asGPU) {
        retval = refit.RefitTrackAsGPU(trk, false, true);
      } else if constexpr (I == mode1asTrackParCov) {
        retval = refit.RefitTrackAsTrackParCov(trk, false, true);
      }
      /*#pragma omp critical
      if (retval > 0) {
        static auto cldump = GPUROOTDump<GPUTPCGMMergedTrack, GPUTPCGMMergedTrack>::getNew("org", "refit", "debugTree");
        cldump.Fill(refit.mPTracks[i], trk);
      }*/
      if (retval > 0) {
        refit.mPTracks[i] = trk;
      } else {
        refit.mPTracks[i].SetOK(false);
      }
    }
  }
}
#if !defined(GPUCA_GPUCODE) || defined(GPUCA_GPUCODE_DEVICE) // FIXME: DR: WORKAROUND to avoid CUDA bug creating host symbols for device code.
template GPUdni() void GPUTrackingRefitKernel::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);
template GPUdni() void GPUTrackingRefitKernel::Thread<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);
#endif
