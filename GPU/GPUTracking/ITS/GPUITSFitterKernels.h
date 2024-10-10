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

/// \file GPUITSFitterKernels.h
/// \author David Rohr, Maximiliano Puccio

#ifndef GPUITSFITTERKERNELS_H
#define GPUITSFITTERKERNELS_H

#include "GPUGeneralKernels.h"
namespace o2::its
{
struct TrackingFrameInfo;
} // namespace o2::its

namespace GPUCA_NAMESPACE::gpu
{
class GPUTPCGMPropagator;
class GPUITSFitter;
class GPUITSTrack;

class GPUITSFitterKernels : public GPUKernelTemplate
{
 public:
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::ITSTracking; }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors);

 protected:
  GPUd() static bool fitTrack(GPUITSFitter& Fitter, GPUTPCGMPropagator& prop, GPUITSTrack& track, int32_t start, int32_t end, int32_t step);
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
