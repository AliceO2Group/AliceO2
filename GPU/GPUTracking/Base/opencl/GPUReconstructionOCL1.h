// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionOCL1.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONOCL1_H
#define GPURECONSTRUCTIONOCL1_H

#include "GPUReconstructionOCL.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const GPUCA_NAMESPACE::gpu::GPUSettingsProcessing& cfg);
#else
extern "C" GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const GPUCA_NAMESPACE::gpu::GPUSettingsProcessing& cfg);
#endif

namespace GPUCA_NAMESPACE::gpu
{
struct GPUReconstructionOCL1Internals;

class GPUReconstructionOCL1Backend : public GPUReconstructionOCL
{
 public:
  ~GPUReconstructionOCL1Backend() override = default;

 protected:
  GPUReconstructionOCL1Backend(const GPUSettingsProcessing& cfg);

  template <class T, int I = 0, typename... Args>
  int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);
  template <class S, class T, int I = 0>
  S& getKernelObject(int num);

  RecoStepField AvailableRecoSteps() override { return (RecoStep::TPCSliceTracking); }
  bool ContextForAllPlatforms() override { return true; }
  bool CheckPlatform(unsigned int i) override;
  int GetOCLPrograms() override;
};

using GPUReconstructionOCL1 = GPUReconstructionKernels<GPUReconstructionOCL1Backend>;
} // namespace GPUCA_NAMESPACE::gpu

#endif
