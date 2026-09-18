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

#include "GPUReconstructionMetal.h"
#include "GPUConstantMem.h"
#include "GPUDefParametersLoad.inc"
#include "GPUReconstructionMetalIncludesHost.h"

#include <cstddef>
#include <map>

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>  // _mh_execute_header

#define GPUErrorReturn(...) \
  {                         \
    GPUError(__VA_ARGS__);  \
    return (1);             \
  }

#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(GPUReconstructionMetalCode_src);

GPUReconstruction* GPUReconstruction_Create_METAL(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionMetal(cfg); }

GPUReconstructionMetal::GPUReconstructionMetal(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionProcessing::KernelInterface<GPUReconstructionMetal, GPUReconstructionDeviceBase>(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionMetalInternals;
    *mParDevice = o2::gpu::internal::GPUDefParametersLoad();
  }
  mDeviceBackendSettings->deviceType = DeviceType::METAL;
}

GPUReconstructionMetal::~GPUReconstructionMetal()
{
  Exit();  // Make sure we destroy everything (in particular the ITS tracker) before we exit
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

int32_t GPUReconstructionMetal::InitDevice_Runtime()
{
  // Propagate processing settings to PoCL runtime.
  // Won't affect other OpenCL runtimes.
  if (int nThreads = mProcessingSettings->nHostThreads; nThreads > 0) {
    auto nThreadsStr = std::to_string(nThreads);
    setenv("PMETAL_CPU_MAX_CU_COUNT", nThreadsStr.c_str(), 1);
  }

  if (mMaster == nullptr) {
    mInternals->device = MTLCreateSystemDefaultDevice();

    int64_t deviceGlobalMem, deviceLocalMem;
    MTLSize deviceMaxWorkGroup = mInternals->device.maxThreadsPerThreadgroup;

    std::string device_name = [mInternals->device.name UTF8String];
    // On Apple Silicon, treat recommended working set as an upper bound
    deviceGlobalMem = mInternals->device.recommendedMaxWorkingSetSize;
    deviceLocalMem = mInternals->device.maxThreadgroupMemoryLength;
    if (GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("Using Metal device %s with properties:", device_name.c_str());
      GPUInfo("\tUnified Memory Architecture = %ld ", mInternals->device.hasUnifiedMemory);
      GPUInfo("\tRecommended Max Working Set = %ld bytes", deviceGlobalMem);
      GPUInfo("\tMax thread group memory = %ld bytes", deviceLocalMem);
      GPUInfo("\tmaxWorkGroup = (%ld, %ld, %ld)", deviceMaxWorkGroup.width, deviceMaxWorkGroup.height, deviceMaxWorkGroup.depth);
      GPUInfo(" ");
    }

    mDeviceName = device_name.c_str();
    // Basically a random number for now.
    mMaxBackendThreads = 1000;

    if (GetMetalPrograms()) {
      return 1;
    }

    if (GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("Metal program and kernels loaded successfully");
    }

    // We only support internal GPUs, so the ownership of the memory is always shared
    mInternals->mem_gpu = [mInternals->device newBufferWithLength:mDeviceMemorySize options:MTLResourceStorageModeShared];
    if (mInternals->mem_gpu == nil) {
      GPUErrorReturn("Metal Memory Allocation Error");
    }

    // We only support internal GPUs, so the ownership of the memory is always shared.
    // FIXME: until I understand how to enable the gGPUConstantMemBufferSize constexpr
    int32_t tmpGPUContantMemBufferSize = 100000;  // gGPUConstantMemBufferSize
    mInternals->mem_constant = [mInternals->device newBufferWithLength:tmpGPUContantMemBufferSize options:MTLResourceStorageModeShared];
    if (mInternals->mem_constant) {
      GPUErrorReturn("Metal Constant Memory Allocation Error");
    }

    for (int32_t i = 0; i < mNStreams; i++) {
      mInternals->commandQueues[i] = [mInternals->device newCommandQueue];
      if (mInternals->commandQueues[i] == nil) {
        GPUErrorReturn("Error creating Metal command queue");
      }
      mInternals->commandBuffers[i] = [mInternals->commandQueues[i] commandBuffer];
      if (mInternals->commandBuffers[i] == nil) {
        GPUErrorReturn("Error creating Metal command buffer");
      }
    }

    mInternals->mem_host = [mInternals->device newBufferWithLength:mHostMemorySize options:MTLResourceStorageModeShared];
    if (mInternals->mem_host == nil) {
      GPUErrorReturn("Error allocating pinned host memory");
    }

    mHostMemoryBase = mInternals->mem_host.contents;
    mHostMemorySize = mInternals->mem_host.allocatedSize;
    mDeviceMemoryBase = mInternals->mem_gpu.contents;
    mDeviceMemorySize = mInternals->mem_gpu.allocatedSize;
    mDeviceConstantMem = (GPUConstantMem*)mInternals->mem_constant.contents;

    if (GetProcessingSettings().debugLevel >= 1) {
      GPUInfo("Memory ptrs: GPU (%ld bytes): %p - Host (%ld bytes): %p", (int64_t)mDeviceMemorySize, mDeviceMemoryBase, (int64_t)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
    }

    GPUInfo("Metal Initialisation successfull");
  } else {
    auto* master = dynamic_cast<GPUReconstructionMetal*>(mMaster);
    mWarpSize = master->mWarpSize;
    mMaxBackendThreads = master->mMaxBackendThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mInternals = master->mInternals;
  }

  for (uint32_t i = 0; i < mEvents.size(); i++) {
    auto* events = (id<MTLSharedEvent>*)mEvents[i].data();
    new (events) id<MTLSharedEvent>[ mEvents[i].size() ];
  }

  return (0);
}

int32_t GPUReconstructionMetal::ExitDevice_Runtime()
{
  // Uninitialize OPENCL
  SynchronizeGPU();

  if (mMaster == nullptr) {
    if (mDeviceMemoryBase) {
      [mInternals->mem_gpu release];
      [mInternals->mem_constant release];
      for (uint32_t i = 0; i < mInternals->functions.size(); i++) {
        [mInternals->functions[i] release];
      }
      mInternals->functions.clear();
    }
    if (mHostMemoryBase) {
      for (int32_t i = 0; i < mNStreams; i++) {
        [mInternals->commandQueues[i] release];
        [mInternals->commandBuffers[i] release];
      }
      [mInternals->mem_host release];
    }

    [mInternals->library release];
    [mInternals->device release];
    GPUInfo("Metal disposed correctly");
  }
  mDeviceMemoryBase = nullptr;
  mHostMemoryBase = nullptr;

  return (0);
}

size_t GPUReconstructionMetal::GPUMemCpy(void* dst, const void* src, size_t sizeBytes, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  if (evList == nullptr) {
    nEvents = 0;
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    stream = -1;
  }

  if (stream == -1) {
    SynchronizeGPU();
  }

  auto realStream = stream == -1 ? 0 : stream;
  id<MTLCommandBuffer> cb = mInternals->commandBuffers[realStream];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  id<MTLBuffer> sourceBuffer = nil;
  id<MTLBuffer> destBuffer = nil;
  ptrdiff_t sourceOffset = 0;
  ptrdiff_t destOffset = 0;

  // Sigh.
  if (src > mHostMemoryBase && src < ((char*)mHostMemoryBase + mHostMemorySize)) {
    sourceBuffer = mInternals->mem_host;
    sourceOffset = (char*)src - (char*)mHostMemoryBase;
  } else if (src > mDeviceMemoryBase && src < ((char*)mDeviceMemoryBase + mDeviceMemorySize)) {
    sourceBuffer = mInternals->mem_gpu;
    sourceOffset = (char*)src - (char*)mDeviceMemoryBase;
  } else {
    GPUErrorReturn("Unknown buffer at %x", src);
  }

  if (dst > mHostMemoryBase && dst < ((char*)mHostMemoryBase + mHostMemorySize)) {
    destBuffer = mInternals->mem_host;
    destOffset = (char*)src - (char*)mHostMemoryBase;
  } else if (dst > mDeviceMemoryBase && dst < ((char*)mDeviceMemoryBase + mDeviceMemorySize)) {
    destBuffer = mInternals->mem_gpu;
    destOffset = (char*)dst - (char*)mDeviceMemoryBase;
  } else {
    GPUErrorReturn("Unknown buffer at %x", src);
  }

  [blit copyFromBuffer:sourceBuffer
          sourceOffset:sourceOffset
              toBuffer:destBuffer
     destinationOffset:destOffset
                  size:sizeBytes];

  [blit endEncoding];
  [cb commit];

  if (GetProcessingSettings().serializeGPU & 2) {
    GPUDebug(("GPUMemCpy " + std::to_string(toGPU)).c_str(), stream, true);
  }
  return sizeBytes;
}

size_t GPUReconstructionMetal::WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream, deviceEvent* ev)
{
  if (stream == -1) {
    SynchronizeGPU();
  }

  auto realStream = stream == -1 ? 0 : stream;
  id<MTLCommandBuffer> cb = mInternals->commandBuffers[realStream];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  id<MTLBuffer> sourceBuffer = nil;
  ptrdiff_t sourceOffset = 0;
  if (src > mHostMemoryBase && src < ((char*)mHostMemoryBase + mHostMemorySize)) {
    sourceBuffer = mInternals->mem_host;
    sourceOffset = (char*)src - (char*)mHostMemoryBase;
  } else if (src > mDeviceMemoryBase && src < ((char*)mDeviceMemoryBase + mDeviceMemorySize)) {
    sourceBuffer = mInternals->mem_gpu;
    sourceOffset = (char*)src - (char*)mDeviceMemoryBase;
  } else {
    GPUErrorReturn("Unknown buffer at %x", src);
  }
  [blit copyFromBuffer:sourceBuffer
          sourceOffset:sourceOffset
              toBuffer:mInternals->mem_constant
     destinationOffset:offset
                  size:size];

  [blit endEncoding];
  [cb commit];

  if (GetProcessingSettings().serializeGPU & 2) {
    GPUDebug("WriteToConstantMemory", stream, true);
  }
  return size;
}

void GPUReconstructionMetal::ReleaseEvent(deviceEvent ev)
{
  // FIXME: is this supposed to reset the event for it to be repurposed
  //        or to decrease the ref count?
  auto mtlEvent = (__bridge id<MTLSharedEvent>)(ev.get<void*>());
  [mtlEvent setSignaledValue:0];
}

void GPUReconstructionMetal::RecordMarker(deviceEvent* ev, int32_t stream)
{
  id<MTLCommandBuffer> cb = mInternals->commandBuffers[stream];
  // Does not change the retain count, so it's important we manage
  // the lifetime of the events outside here.
  auto mtlEvent = (__bridge id<MTLSharedEvent>)(ev->get<void*>());
  [cb encodeSignalEvent:mtlEvent value:1];
  [cb commit];
}

int32_t GPUReconstructionMetal::DoStuckProtection(int32_t stream, deviceEvent event)
{
  if (GetProcessingSettings().stuckProtection) {
    GPUError("Stuck protection not implemented for Metal");
  } else {
    [mInternals->commandBuffers[stream] waitUntilCompleted];
  }
  return 0;
}

void GPUReconstructionMetal::SynchronizeGPU()
{
  for (int32_t i = 0; i < mNStreams; i++) {
    [mInternals->commandBuffers[i] waitUntilCompleted];
  }
}

void GPUReconstructionMetal::SynchronizeStream(int32_t stream)
{
  [mInternals->commandBuffers[stream] waitUntilCompleted];
}

void GPUReconstructionMetal::SynchronizeEvents(deviceEvent* evList, int32_t nEvents)
{
  // I wait for everything to complete for now...
  for (int32_t si = 0; si < mNStreams; si++) {
    id<MTLCommandBuffer> cb = mInternals->commandBuffers[si];
    [cb waitUntilCompleted];
  }
}

void GPUReconstructionMetal::StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents)
{
  // Encode commands to wait for all the events
  id<MTLCommandBuffer> cb = mInternals->commandBuffers[stream];
  for (int32_t ei = 0; ei < nEvents; ei++) {
    auto mtlEvent = (__bridge id<MTLSharedEvent>)(evList[ei].get<void*>());
    [cb encodeWaitForEvent:mtlEvent value:1];
  }
  [cb commit];
  [cb waitUntilCompleted];
}

bool GPUReconstructionMetal::IsEventDone(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    auto mtlEvent = (__bridge id<MTLSharedEvent>)(evList[i].get<void*>());
    if (mtlEvent.signaledValue == 0) {
      return false;
    }
  }
  return true;
}

int32_t GPUReconstructionMetal::GPUDebug(const char* state, int32_t stream, bool force)
{
  // Wait for Metal-Kernel to finish and check for Metal errors afterwards, in case of debugmode
  if (!force && GetProcessingSettings().debugLevel <= 0) {
    return (0);
  }
  for (int32_t si = 0; si < mNStreams; si++) {
    [mInternals->commandBuffers[si] waitUntilCompleted];
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

int32_t GPUReconstructionMetal::GPUChkErrInternal(const int64_t error, const char* file, int32_t line) const
{
  // Not sure how metal returns errors.
  if (error != 0) {
    GPUError("Metal Error: %ld / %s (%s:%d)", error, "Unknown", file, line);
  }
  return error != 0;
}

// Return pointer+size for (__DATA|__DATA_CONST, "__gpu_resource") from the image
// that matches `image_name_substr` (e.g. "libO2GPUReconstruction.dylib").
static const uint8_t* find_gpu_resource_in_image(const char* image_name_substr,
                                                 unsigned long* out_size)
{
  uint32_t count = _dyld_image_count();
  for (uint32_t i = 0; i < count; ++i) {
    const char* name = _dyld_get_image_name(i);
    if (!name || !strstr(name, image_name_substr)) {
      continue;
    }

    const struct mach_header* mh = _dyld_get_image_header(i);

    const auto* mh64 = (const struct mach_header_64*)mh;
    const auto* p = (const uint8_t*)
      getsectiondata(mh64, "__DATA", "__gpu_resource", out_size);
    if (!p) {
      p = (const uint8_t*)
        getsectiondata(mh64, "__DATA_CONST", "__gpu_resource", out_size);
    }
    if (p) {
      return p;
    }
  }
  return nullptr;
}

int32_t GPUReconstructionMetal::GetMetalPrograms()
{
  // No need for now...
  [[maybe_unused]] const char* metalBuildFlags = GetProcessingSettings().metalOverrideSourceBuildFlags != "" ? GetProcessingSettings().metalOverrideSourceBuildFlags.c_str() : GPUCA_M_STR(GPUCA_METAL_BUILD_FLAGS);

  GPUInfo("Compiling Metal program from sources (Platform version %s)", [mInternals->device.architecture.name cStringUsingEncoding:NSUTF8StringEncoding]);

  unsigned long sz = 0;
  const char* p = (char const*)find_gpu_resource_in_image("libO2GPUTrackingMETAL.dylib", &sz);
  auto source = [[NSString alloc] initWithCString:p encoding:NSUTF8StringEncoding];

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

  // Equivalent to clCreateProgramWithSource
  mInternals->library = [mInternals->device newLibraryWithSource:source
                                                         options:options
                                                           error:&error];

  if (error != nil) {
    NSLog(@"%@", error);
    NSLog(@"Error dump:\n%@", [error description]);
    NSLog(@"Error debug dump:\n%@", [error debugDescription]);
    GPUError("Error creating Metal program from binary");
    return 1;
  }

  return AddKernels();
}
