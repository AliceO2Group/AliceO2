// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <Foundation/Foundation.h>
#include "GPUReconstructionKernelIncludes.h"
#include "GPUReconstructionMetalIncludesHost.h"

#include "GPUReconstructionMetalKernelsSpecialize.inc"
#include "GPUReconstructionProcessingKernels.inc"

template void GPUReconstructionProcessing::KernelInterface<GPUReconstructionMetal, GPUReconstructionDeviceBase>::runKernelVirtual(const int num, const void* args);

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionMetal::runKernelBackend(const krnlSetupTime& _xyz, const Args&... args)
{
  id<MTLFunction> function = mInternals->functions[GetKernelNum<T, I>()];
  auto& kExec = _xyz.x;
  auto& runRange = _xyz.y;
  // auto& events = _xyz.z;
  // auto& t = _xyz.t;

  NSError* error = nil;
  auto pso = [mInternals->device newComputePipelineStateWithFunction:function error:&error];
  id<MTLComputeCommandEncoder> computeEncoder = [mInternals->commandBuffers[kExec.stream] computeCommandEncoder];

  // Map buffers and states
  [computeEncoder setComputePipelineState:pso];
  [computeEncoder setBuffer:mInternals->mem_gpu offset:0 atIndex:0];
  [computeEncoder setBuffer:mInternals->mem_constant offset:0 atIndex:1];
  [computeEncoder setBuffer:mInternals->mem_host offset:0 atIndex:2];

  MTLSize gridSize = MTLSizeMake(runRange.index, 1, 1);

  NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
  if (threadGroupSize > runRange.index) {
    threadGroupSize = runRange.index;
  }

  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
  [computeEncoder dispatchThreads:gridSize
            threadsPerThreadgroup:threadgroupSize];
}

template <class T, int32_t I>
int32_t GPUReconstructionMetal::AddKernel()
{
  NSString* kname = [[NSString alloc] initWithFormat:@"krnl_%s", GetKernelName<T, I>()];

  id<MTLFunction> krnl = [mInternals->library newFunctionWithName:kname];
  if (krnl == nil) {
    GPUError("Error creating Metal Kernel: %s", [kname cStringUsingEncoding:NSUTF8StringEncoding]);
    return 1;
  }

  mInternals->functions.emplace_back(krnl);
  return 0;
}

template <class S, class T, int32_t I>
S& GPUReconstructionMetal::getKernelObject()
{
  return mInternals->functions[GetKernelNum<T, I>()];
}

int32_t GPUReconstructionMetal::AddKernels()
{
#define GPUCA_KRNL(x_class, ...)                     \
  if (AddKernel<GPUCA_M_KRNL_TEMPLATE(x_class)>()) { \
    return 1;                                        \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
  return 0;
}
