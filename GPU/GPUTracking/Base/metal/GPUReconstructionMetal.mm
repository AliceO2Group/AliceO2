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

#include "GPUConstantMem.h"
#include "GPUDefParametersLoad.inc"
#include "GPUReconstructionMetal.h"

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

    // FIXME: until I understand how to enable the gGPUConstantMemBufferSize constexpr
    int32_t tmpGPUConstantMemBufferSize = 100000;  // gGPUConstantMemBufferSize
    mInternals->mem_constant = [mInternals->device newBufferWithLength:tmpGPUConstantMemBufferSize options:MTLResourceStorageModeShared];
    if (mInternals->mem_constant == nil) {
      GPUErrorReturn("Metal Constant Memory Allocation Error");
    }

    for (int32_t i = 0; i < mNStreams; i++) {
      mInternals->commandQueues[i] = [mInternals->device newCommandQueue];
      if (mInternals->commandQueues[i] == nil) {
        GPUErrorReturn("Error creating Metal command queue");
      }
      mInternals->commandBuffers[i] = [[mInternals->commandQueues[i] commandBuffer] retain];
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
    // The kernels reconstruct device pointers from these bases, so they must be
    // the addresses the GPU sees, not the CPU-visible ones of a shared buffer.
    mDeviceMemoryBase = (void*)(uintptr_t)mInternals->mem_gpu.gpuAddress;
    mDeviceMemorySize = mInternals->mem_gpu.allocatedSize;
    mDeviceConstantMem = (GPUConstantMem*)(uintptr_t)mInternals->mem_constant.gpuAddress;

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
  // Tear down Metal
  SynchronizeGPU();

  if (mMaster == nullptr) {
    if (mDeviceMemoryBase) {
      [mInternals->mem_gpu release];
      [mInternals->mem_constant release];
      for (uint32_t i = 0; i < mInternals->functions.size(); i++) {
        [mInternals->functions[i] release];
      }
      mInternals->functions.clear();
      for (uint32_t i = 0; i < mInternals->pipelines.size(); i++) {
        [mInternals->pipelines[i] release];
      }
      mInternals->pipelines.clear();
      mInternals->maxThreadsPerGroup.clear();
    }
    if (mHostMemoryBase) {
      for (int32_t i = 0; i < mNStreams; i++) {
        DrainStream(i);
        [mInternals->commandBuffers[i] release];
        [mInternals->commandQueues[i] release];
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

namespace
{
// Where a pointer lives as far as the backend is concerned. mDeviceMemoryBase is
// a GPU address, so the CPU-visible address of owned memory has to be taken from
// the buffer rather than from the pointer itself.
struct MetalMemRef {
  id<MTLBuffer> buffer;  // nil for memory the backend does not own
  size_t offset;
  char* cpu;
};

MetalMemRef metalResolve(const GPUReconstructionMetalInternals* internals, const void* p, const void* hostBase, size_t hostSize, const void* devBase, size_t devSize)
{
  if (p >= hostBase && p < ((const char*)hostBase + hostSize)) {
    const size_t o = (const char*)p - (const char*)hostBase;
    return {internals->mem_host, o, (char*)internals->mem_host.contents + o};
  }
  if (p >= devBase && p < ((const char*)devBase + devSize)) {
    const size_t o = (const char*)p - (const char*)devBase;
    return {internals->mem_gpu, o, (char*)internals->mem_gpu.contents + o};
  }
  return {nil, 0, (char*)p};
}
}  // namespace

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
  const MetalMemRef s = metalResolve(mInternals, src, mHostMemoryBase, mHostMemorySize, mDeviceMemoryBase, mDeviceMemorySize);
  const MetalMemRef d = metalResolve(mInternals, dst, mHostMemoryBase, mHostMemorySize, mDeviceMemoryBase, mDeviceMemorySize);

  if (s.buffer == nil || d.buffer == nil) {
    // A blit moves data between MTLBuffers only, and the calibration objects
    // live in ordinary memory. Both pools are shared storage on a unified
    // architecture, so the copy can be done on the host once the stream is idle.
    DrainStream(realStream);
    memcpy(d.cpu, s.cpu, sizeBytes);
  } else {
    id<MTLCommandBuffer> cb = mInternals->commandBuffers[realStream];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:s.buffer
            sourceOffset:s.offset
                toBuffer:d.buffer
       destinationOffset:d.offset
                    size:sizeBytes];
    [blit endEncoding];
    CommitStream(realStream);
  }

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
  const MetalMemRef s = metalResolve(mInternals, src, mHostMemoryBase, mHostMemorySize, mDeviceMemoryBase, mDeviceMemorySize);

  if (s.buffer == nil) {
    DrainStream(realStream);
    memcpy((char*)mInternals->mem_constant.contents + offset, s.cpu, size);
  } else {
    id<MTLCommandBuffer> cb = mInternals->commandBuffers[realStream];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:s.buffer
            sourceOffset:s.offset
                toBuffer:mInternals->mem_constant
       destinationOffset:offset
                    size:size];
    [blit endEncoding];
    CommitStream(realStream);
  }

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
  CommitStream(stream);
}

void GPUReconstructionMetal::CommitStream(int32_t stream)
{
  id<MTLCommandBuffer> cb = mInternals->commandBuffers[stream];
  [cb commit];
  // Hand the committed buffer over to the stream's submission list, which owns
  // it until it is waited for, and open a fresh one to encode the next work in.
  mInternals->submittedBuffers[stream].emplace_back(cb);
  mInternals->commandBuffers[stream] = [[mInternals->commandQueues[stream] commandBuffer] retain];
}

void GPUReconstructionMetal::DrainStream(int32_t stream)
{
  // Waiting on the buffer that is currently open would block forever: it is not
  // committed, so it never completes. Only what was submitted can be waited for.
  for (id<MTLCommandBuffer> cb : mInternals->submittedBuffers[stream]) {
    [cb waitUntilCompleted];
    [cb release];
  }
  mInternals->submittedBuffers[stream].clear();
}

void GPUReconstructionMetal::SynchronizeGPU()
{
  for (int32_t i = 0; i < mNStreams; i++) {
    DrainStream(i);
  }
}

void GPUReconstructionMetal::SynchronizeStream(int32_t stream)
{
  DrainStream(stream);
}

void GPUReconstructionMetal::SynchronizeEvents(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t si = 0; si < mNStreams; si++) {
    DrainStream(si);
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
  CommitStream(stream);
  DrainStream(stream);
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
    DrainStream(si);
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
  GPUInfo("Loading Metal library (Platform version %s)", [mInternals->device.architecture.name cStringUsingEncoding:NSUTF8StringEncoding]);

  unsigned long sz = 0;
  const uint8_t* p = find_gpu_resource_in_image("libO2GPUTrackingMETAL.dylib", &sz);
  if (p == nullptr || sz == 0) {
    GPUError("Metal library not found in the __gpu_resource section");
    return 1;
  }

  // the section is part of the mapped image, so it outlives the dispatch_data_t
  // and does not have to be copied
  dispatch_data_t blob = dispatch_data_create(p, sz, nullptr, ^{
                                                     });

  NSError* error = nil;
  mInternals->library = [mInternals->device newLibraryWithData:blob error:&error];

  if (mInternals->library == nil) {
    NSLog(@"%@", error);
    GPUError("Error loading the Metal library");
    return 1;
  }

  return AddKernels();
}
