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

/// \file GPUReconstructionCPUKernels.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONICPUKERNELS_H
#define GPURECONSTRUCTIONICPUKERNELS_H

#include "GPUReconstructionCPU.h"
#include "GPUSettings.h"
#include "GPULogging.h"

namespace o2::gpu
{

template <class S, int32_t I, typename... Args>
inline void GPUReconstructionCPU::runKernelInterface(krnlSetup&& setup, Args const&... args)
{
  HighResTimer* t = nullptr;
  gpudatatypes::RecoStep myStep = S::GetRecoStep() == gpudatatypes::RecoStep::NoRecoStep ? setup.x.step : S::GetRecoStep();
  if (myStep == gpudatatypes::RecoStep::NoRecoStep) {
    throw std::runtime_error("Failure running general kernel without defining RecoStep");
  }
  int32_t cpuFallback = IsGPU() ? (setup.x.device == krnlDeviceType::CPU ? 2 : (mRecoSteps.stepsGPUMask & myStep) != myStep) : 0;
  uint32_t& nThreads = setup.x.nThreads;
  uint32_t& nBlocks = setup.x.nBlocks;
  const uint32_t stream = setup.x.stream;
  auto prop = getKernelProperties<S, I>();
  const int32_t autoThreads = cpuFallback ? 1 : prop.nThreads;
  const int32_t autoBlocks = cpuFallback ? 1 : (prop.forceBlocks ? prop.forceBlocks : (prop.minBlocks * mMultiprocessorCount));
  if (nBlocks == (uint32_t)-1) {
    nBlocks = (nThreads + autoThreads - 1) / autoThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (uint32_t)-2) {
    nBlocks = nThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (uint32_t)-3) {
    nBlocks = autoBlocks;
    nThreads = autoThreads;
  } else if ((int32_t)nThreads < 0) {
    nThreads = cpuFallback ? 1 : -nThreads;
  }
  if (nThreads > GPUCA_MAX_THREADS) {
    throw std::runtime_error("GPUCA_MAX_THREADS exceeded");
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("Running kernel %s (Stream %d, Index %d, Grid %d/%d) on %s", GetKernelName<S, I>(), stream, setup.y.index, nBlocks, nThreads, cpuFallback == 2 ? "CPU (forced)" : (cpuFallback ? "CPU (fallback)" : mDeviceName.c_str()));
  }
  if (nThreads == 0 || nBlocks == 0) {
    return;
  }
  if (GetProcessingSettings().debugLevel >= 1) {
    t = &getKernelTimer<S, I>(myStep, !IsGPU() || cpuFallback ? getHostThreadIndex() : stream);
    if ((!GetProcessingSettings().deviceTimers || !IsGPU() || cpuFallback) && (mNActiveThreadsOuterLoop < 2 || getHostThreadIndex() == 0)) {
      t->Start();
    }
  }
  double deviceTimerTime = 0.;
  krnlSetupArgs<S, I, Args...> argPack{{}, {{setup.x, setup.y, setup.z}, deviceTimerTime}, {args...}};
  const uint32_t num = GetKernelNum<S, I>();
  if (cpuFallback) {
    GPUReconstructionCPU::runKernelVirtual(num, &argPack);
  } else {
    runKernelVirtual(num, &argPack);
  }

  if (GPUDebug(GetKernelName<S, I>(), stream, GetProcessingSettings().serializeGPU & 1)) {
    throw std::runtime_error("kernel failure");
  }
  if (GetProcessingSettings().debugLevel >= 1) {
    if (t) {
      if (deviceTimerTime != 0.) {
        t->AddTime(deviceTimerTime);
        if (t->IsRunning()) {
          t->Abort();
        }
      } else if (t->IsRunning()) {
        t->Stop();
      }
    }
    if (CheckErrorCodes(cpuFallback) && !GetProcessingSettings().ignoreNonFatalGPUErrors) {
      throw std::runtime_error("kernel error code");
    }
  }
}

} // namespace o2::gpu

#endif
