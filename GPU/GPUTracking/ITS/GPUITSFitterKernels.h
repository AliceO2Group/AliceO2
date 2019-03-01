// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUITSFitterKernels.h
/// \author David Rohr, Maximiliano Puccio

#ifndef GPUITSFITTERKERNELS_H
#define GPUITSFITTERKERNELS_H

class GPUTPCGMPropagator;
class GPUITSFitter;

#include "GPUGeneralKernels.h"
namespace o2
{
namespace ITS
{
class GPUITSTrack;
struct TrackingFrameInfo;
} // namespace ITS
} // namespace o2

class GPUITSFitterKernel : public GPUKernelTemplate
{
 public:
  GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::ITSTracking; }
#if defined(GPUCA_BUILD_ITS)
  template <int iKernel = 0>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, workerType& workers);

 protected:
  GPUd() static bool fitTrack(GPUITSFitter& Fitter, GPUTPCGMPropagator& prop, o2::ITS::GPUITSTrack& track, int start, int end, int step);
#endif
};

#endif
