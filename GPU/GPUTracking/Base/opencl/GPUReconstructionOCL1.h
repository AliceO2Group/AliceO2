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

/// \file GPUReconstructionOCL1.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONOCL1_H
#define GPURECONSTRUCTIONOCL1_H

#include "GPUReconstructionOCL.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace GPUCA_NAMESPACE::gpu
{
struct GPUReconstructionOCL1Internals;

class GPUReconstructionOCL1Backend : public GPUReconstructionOCL
{
 public:
  ~GPUReconstructionOCL1Backend() override = default;

 protected:
  GPUReconstructionOCL1Backend(const GPUSettingsDeviceBackend& cfg);

  template <class T, int I = 0, typename... Args>
  int runKernelBackend(const krnlSetupArgs<T, I, Args...>& args);
  template <class S, class T, int I, bool MULTI>
  S& getKernelObject();

  RecoStepField AvailableGPURecoSteps() override { return (RecoStep::TPCSliceTracking); }
  bool ContextForAllPlatforms() override { return true; }
  bool CheckPlatform(unsigned int i) override;
  int GetOCLPrograms() override;
};

using GPUReconstructionOCL1 = GPUReconstructionKernels<GPUReconstructionOCL1Backend>;
} // namespace GPUCA_NAMESPACE::gpu

#endif
