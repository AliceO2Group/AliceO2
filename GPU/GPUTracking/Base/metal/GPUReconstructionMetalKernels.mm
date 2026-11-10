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

#include "GPUReconstructionMetalIncludesHost.h"  // must come first

#include <Foundation/Foundation.h>
#include "GPUReconstructionKernelIncludes.h"

#include "GPUReconstructionMetalKernelsSpecialize.inc"
#include "GPUReconstructionProcessingKernels.inc"

template void GPUReconstructionProcessing::KernelInterface<GPUReconstructionMetal, GPUReconstructionDeviceBase>::runKernelVirtual(const int num, const void* args);

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionMetal::runKernelBackend(const krnlSetupTime& _xyz, const Args&... args)
{
  const uint32_t num = GetKernelNum<T, I>();
  auto& kExec = _xyz.x;
  auto& runRange = _xyz.y;
  auto& kEvents = _xyz.z;
  if (kEvents.evList) {
    StreamWaitForEvents(kExec.stream, kEvents.evList, kEvents.nEvents);
  }
  // Device timers: the kernel gets a command buffer of its own, whose GPU time is the kernel's
  const bool timed = GetProcessingSettings().deviceTimers && GetProcessingSettings().debugLevel > 0;
  if (timed && mInternals->streamHasWork[kExec.stream]) {
    CommitStream(kExec.stream);
  }

  id<MTLComputePipelineState> pso = mInternals->pipelines[num];
  id<MTLComputeCommandEncoder> computeEncoder = [mInternals->commandBuffers[kExec.stream] computeCommandEncoder];
  metalBeginEncoder(mInternals, computeEncoder, kExec.stream);

  // Slots 0 to 2 are what GPUCA_KRNL_BUFFERS and GPUCA_KRNL_SECTOR_ARG declare
  // in the generated signature; the kernel's own arguments follow from slot 3.
  [computeEncoder setComputePipelineState:pso];
  [computeEncoder setBuffer:mInternals->mem_gpu offset:0 atIndex:0];
  [computeEncoder setBuffer:mInternals->mem_constant offset:0 atIndex:1];
  const int32_t iSector = runRange.index;
  [computeEncoder setBytes:&iSector length:sizeof(iSector) atIndex:2];
  NSUInteger argIndex = 3;
  ((void)[&](const auto& arg) { [computeEncoder setBytes:&arg length:sizeof(arg) atIndex:argIndex++]; }(args), ...);

  // runKernelInterface has already resolved the grid, so nBlocks and nThreads
  // carry the tuned launch bounds and map onto Metal's threadgroups directly.
  NSUInteger threadGroupSize = kExec.nThreads;
  if (threadGroupSize > mInternals->maxThreadsPerGroup[num]) {
    threadGroupSize = mInternals->maxThreadsPerGroup[num];
  }
  if (mInternals->timeline) {
    mInternals->timelineOps[kExec.stream] += std::string(" k:") + GetKernelName<T, I>() + "/" + std::to_string(kExec.nBlocks);
    mInternals->timelineKernels[kExec.stream]++;
  }
  [computeEncoder dispatchThreadgroups:MTLSizeMake(kExec.nBlocks, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
  metalEndEncoder(mInternals, computeEncoder, kExec.stream);
  id<MTLCommandBuffer> cb = [mInternals->commandBuffers[kExec.stream] retain];
  if (kEvents.ev) {
    RecordMarker(kEvents.ev, kExec.stream);
  } else if (mInternals->maxKernelsPerBuffer && ++mInternals->streamKernels[kExec.stream] >= mInternals->maxKernelsPerBuffer) {
    CommitStream(kExec.stream);
  }
  if (timed) {
    if (mInternals->commandBuffers[kExec.stream] == cb) {
      CommitStream(kExec.stream);
    }
    DrainStream(kExec.stream);
    [cb waitUntilCompleted];  // the times are only set once the completion is processed
    _xyz.t = cb.GPUEndTime - cb.GPUStartTime;
  }
  [cb release];
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

  // The kernel sizes its threadgroup memory from the tuning table at compile
  // time, so a launch must not exceed what it was built for either way.
  const uint32_t tunedThreads = this->template getKernelProperties<T, I>().nThreads;

  // Like __launch_bounds__: without it the register allocation may leave room
  // for fewer threads than the kernel is tuned for.
  MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
  desc.computeFunction = krnl;
  desc.maxTotalThreadsPerThreadgroup = tunedThreads;
  NSError* error = nil;
  id<MTLComputePipelineState> pso = [mInternals->device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
  [desc release];
  if (pso == nil) {
    NSLog(@"%@", error);
    GPUError("Error creating Metal pipeline state: %s", [kname cStringUsingEncoding:NSUTF8StringEncoding]);
    return 1;
  }

  if (tunedThreads > pso.maxTotalThreadsPerThreadgroup) {
    // Fewer threads than the kernel was built for: safe for the shared arrays,
    // but the scan and compaction kernels assume every slot has an owner.
    GPUError("Metal kernel %s is tuned for %u threads but this device allows only %lu per threadgroup", GetKernelName<T, I>(), tunedThreads, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
  }

  mInternals->functions.emplace_back(krnl);
  mInternals->pipelines.emplace_back(pso);
  mInternals->maxThreadsPerGroup.emplace_back(pso.maxTotalThreadsPerThreadgroup);
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
