// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingRefitKernel.cxx
/// \author David Rohr

#include "GPUTrackingRefitKernel.h"
#include "GPUTrackingRefit.h"

using namespace GPUCA_NAMESPACE::gpu;

template <int I>
GPUdii() void GPUTrackingRefitKernel::Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  auto& refit = processors.trackingRefit;
  for (unsigned int i = get_global_id(0); i < processors.ioPtrs.nMergedTracks; i += get_global_size(0)) {
    if (refit.mPTracks[i].OK()) {
      GPUTPCGMMergedTrack trk = refit.mPTracks[i];
      int retval;
      if constexpr (I == mode0asGPU) {
        retval = refit.RefitTrackAsGPU(trk, false, true);
      } else if constexpr (I == mode1asTrackParCov) {
        retval = refit.RefitTrackAsTrackParCov(trk, false, true);
      }
      if (retval > 0) {
        refit.mPTracks[i] = trk;
      } else {
        refit.mPTracks[i].SetOK(false);
      }
    }
  }
}
template GPUd() void GPUTrackingRefitKernel::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors);
template GPUd() void GPUTrackingRefitKernel::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors);
