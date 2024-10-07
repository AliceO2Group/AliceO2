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

/// \file GPUReconstructionOCL2.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONOCL2_H
#define GPURECONSTRUCTIONOCL2_H

#include "GPUReconstructionOCL.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL2(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL2(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace GPUCA_NAMESPACE::gpu
{
struct GPUReconstructionOCL2Internals;

class GPUReconstructionOCL2Backend : public GPUReconstructionOCL
{
 public:
  ~GPUReconstructionOCL2Backend() override = default;

 protected:
  GPUReconstructionOCL2Backend(const GPUSettingsDeviceBackend& cfg);

  template <class T, int32_t I = 0, typename... Args>
  int32_t runKernelBackend(const krnlSetupArgs<T, I, Args...>& args);
  template <class S, class T, int32_t I, bool MULTI>
  S& getKernelObject();

  int32_t GetOCLPrograms() override;
  bool CheckPlatform(uint32_t i) override;
};

using GPUReconstructionOCL2 = GPUReconstructionKernels<GPUReconstructionOCL2Backend>;
} // namespace GPUCA_NAMESPACE::gpu

#endif
