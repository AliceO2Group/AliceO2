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

/// \file GPUReconstructionHIP.hip.cxx
/// \author David Rohr

#define GPUCA_GPUCODE_HOSTONLY
#include "GPUReconstructionHIPIncludes.h"

#include "GPUDef.h"

#include "GPUReconstructionHIP.h"
#include "GPUReconstructionHIPInternals.h"
#include "HIPThrustHelpers.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

__global__ void dummyInitKernel(void*) {}

#include "GPUReconstructionIncludesITS.h"

GPUReconstructionHIPBackend::GPUReconstructionHIPBackend(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionDeviceBase(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionHIPInternals;
  }
}

GPUReconstructionHIPBackend::~GPUReconstructionHIPBackend()
{
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

int GPUReconstructionHIPBackend::GPUFailedMsgAI(const long long int error, const char* file, int line)
{
  // Check for HIP Error and in the case of an error display the corresponding error string
  if (error == hipSuccess) {
    return (0);
  }
  GPUError("HIP Error: %lld / %s (%s:%d)", error, hipGetErrorString((hipError_t)error), file, line);
  return 1;
}

void GPUReconstructionHIPBackend::GPUFailedMsgA(const long long int error, const char* file, int line)
{
  if (GPUFailedMsgAI(error, file, line)) {
    static bool runningCallbacks = false;
    if (IsInitialized() && runningCallbacks == false) {
      runningCallbacks = true;
      CheckErrorCodes(false, true);
    }
    throw std::runtime_error("HIP Failure");
  }
}

GPUReconstructionHIP::GPUReconstructionHIP(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionKernels(cfg)
{
  mDeviceBackendSettings.deviceType = DeviceType::HIP;
}

GPUReconstructionHIP::~GPUReconstructionHIP()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit
}

GPUReconstruction* GPUReconstruction_Create_HIP(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionHIP(cfg); }

void GPUReconstructionHIP::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame>* timeFrame)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraitsGPU);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraitsGPU);
  }
  if (timeFrame) {
    timeFrame->reset(new o2::its::gpu::TimeFrameGPU);
  }
}

void GPUReconstructionHIP::UpdateAutomaticProcessingSettings()
{
  GPUCA_GPUReconstructionUpdateDefaults();
}

int GPUReconstructionHIP::InitDevice_Runtime()
{
  if (mMaster == nullptr) {
    hipDeviceProp_t hipDeviceProp;
    int count, bestDevice = -1;
    double bestDeviceSpeed = -1, deviceSpeed;
    if (GPUFailedMsgI(hipGetDeviceCount(&count))) {
      GPUError("Error getting HIP Device Count");
      return (1);
    }
    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Available HIP devices:");
    }
    std::vector<bool> devicesOK(count, false);
    std::vector<size_t> devMemory(count, 0);
    bool contextCreated = false;
    for (int i = 0; i < count; i++) {
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Examining device %d", i);
      }
      size_t free, total;
      if (GPUFailedMsgI(hipSetDevice(i))) {
        if (mProcessingSettings.debugLevel >= 4) {
          GPUWarning("Couldn't create context for device %d. Skipping it.", i);
        }
        continue;
      }
      contextCreated = true;
      if (GPUFailedMsgI(hipMemGetInfo(&free, &total))) {
        if (mProcessingSettings.debugLevel >= 4) {
          GPUWarning("Error obtaining HIP memory info about device %d! Skipping it.", i);
        }
        GPUFailedMsg(hipDeviceReset());
        continue;
      }
      if (count > 1) {
        GPUFailedMsg(hipDeviceReset());
        contextCreated = false;
      }
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Obtained current memory usage for device %d", i);
      }
      if (GPUFailedMsgI(hipGetDeviceProperties(&hipDeviceProp, i))) {
        continue;
      }
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Obtained device properties for device %d", i);
      }
      int deviceOK = true;
      const char* deviceFailure = "";

      deviceSpeed = (double)hipDeviceProp.multiProcessorCount * (double)hipDeviceProp.clockRate * (double)hipDeviceProp.warpSize * (double)hipDeviceProp.major * (double)hipDeviceProp.major;
      if (mProcessingSettings.debugLevel >= 2) {
        GPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem %lld)%s %s", deviceOK ? " " : "[", i, hipDeviceProp.name, hipDeviceProp.major, hipDeviceProp.minor, (long long int)hipDeviceProp.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
      }
      if (!deviceOK) {
        continue;
      }
      devicesOK[i] = true;
      if (deviceSpeed > bestDeviceSpeed) {
        bestDevice = i;
        bestDeviceSpeed = deviceSpeed;
      } else {
        if (mProcessingSettings.debugLevel >= 2 && mProcessingSettings.deviceNum < 0) {
          GPUInfo("Skipping: Speed %f < %f\n", deviceSpeed, bestDeviceSpeed);
        }
      }
    }
    bool noDevice = false;
    if (bestDevice == -1) {
      GPUWarning("No %sHIP Device available, aborting HIP Initialisation (Required mem: %lld)", count ? "appropriate " : "", (long long int)mDeviceMemorySize);
      noDevice = true;
    } else if (mProcessingSettings.deviceNum > -1) {
      if (mProcessingSettings.deviceNum >= (signed)count) {
        GPUWarning("Requested device ID %d does not exist", mProcessingSettings.deviceNum);
        noDevice = true;
      } else if (!devicesOK[mProcessingSettings.deviceNum]) {
        GPUWarning("Unsupported device requested (%d)", mProcessingSettings.deviceNum);
        noDevice = true;
      } else {
        bestDevice = mProcessingSettings.deviceNum;
      }
    }
    if (noDevice) {
      if (contextCreated) {
        GPUFailedMsgI(hipDeviceReset());
      }
      return (1);
    }
    mDeviceId = bestDevice;

    GPUFailedMsgI(hipGetDeviceProperties(&hipDeviceProp, mDeviceId));

    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Using HIP Device %s with Properties:", hipDeviceProp.name);
      GPUInfo("\ttotalGlobalMem = %lld", (unsigned long long int)hipDeviceProp.totalGlobalMem);
      GPUInfo("\tsharedMemPerBlock = %lld", (unsigned long long int)hipDeviceProp.sharedMemPerBlock);
      GPUInfo("\tregsPerBlock = %d", hipDeviceProp.regsPerBlock);
      GPUInfo("\twarpSize = %d", hipDeviceProp.warpSize);
      GPUInfo("\tmaxThreadsPerBlock = %d", hipDeviceProp.maxThreadsPerBlock);
      GPUInfo("\tmaxThreadsDim = %d %d %d", hipDeviceProp.maxThreadsDim[0], hipDeviceProp.maxThreadsDim[1], hipDeviceProp.maxThreadsDim[2]);
      GPUInfo("\tmaxGridSize = %d %d %d", hipDeviceProp.maxGridSize[0], hipDeviceProp.maxGridSize[1], hipDeviceProp.maxGridSize[2]);
      GPUInfo("\ttotalConstMem = %lld", (unsigned long long int)hipDeviceProp.totalConstMem);
      GPUInfo("\tmajor = %d", hipDeviceProp.major);
      GPUInfo("\tminor = %d", hipDeviceProp.minor);
      GPUInfo("\tclockRate = %d", hipDeviceProp.clockRate);
      GPUInfo("\tmemoryClockRate = %d", hipDeviceProp.memoryClockRate);
      GPUInfo("\tmultiProcessorCount = %d", hipDeviceProp.multiProcessorCount);
      GPUInfo(" ");
    }
    if (hipDeviceProp.warpSize != GPUCA_WARP_SIZE) {
      throw std::runtime_error("Invalid warp size on GPU");
    }
    mBlockCount = hipDeviceProp.multiProcessorCount;
    mMaxThreads = std::max<int>(mMaxThreads, hipDeviceProp.maxThreadsPerBlock * mBlockCount);
    mWarpSize = 64;
    mDeviceName = hipDeviceProp.name;
    mDeviceName += " (HIP GPU)";

    if (hipDeviceProp.major < 3) {
      GPUError("Unsupported HIP Device");
      return (1);
    }
#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (gGPUConstantMemBufferSize > hipDeviceProp.totalConstMem) {
      GPUError("Insufficient constant memory available on GPU %d < %d!", (int)hipDeviceProp.totalConstMem, (int)gGPUConstantMemBufferSize);
      return (1);
    }
#endif

    if (contextCreated == 0 && GPUFailedMsgI(hipSetDevice(mDeviceId))) {
      GPUError("Could not set HIP Device!");
      return (1);
    }
    if (GPUFailedMsgI(hipSetDeviceFlags(hipDeviceScheduleBlockingSync))) {
      GPUError("Could not set HIP Device!");
      return (1);
    }

    /*if (GPUFailedMsgI(hipDeviceSetLimit(hipLimitStackSize, GPUCA_GPU_STACK_SIZE)))
    {
      GPUError("Error setting HIP stack size");
      GPUFailedMsgI(hipDeviceReset());
      return(1);
    }*/

    if (mDeviceMemorySize > hipDeviceProp.totalGlobalMem || GPUFailedMsgI(hipMalloc(&mDeviceMemoryBase, mDeviceMemorySize))) {
      size_t free, total;
      GPUFailedMsg(hipMemGetInfo(&free, &total));
      GPUError("HIP Memory Allocation Error (trying %lld bytes, %lld available on GPU, %lld free)", (long long int)mDeviceMemorySize, (long long int)hipDeviceProp.totalGlobalMem, (long long int)free);
      GPUFailedMsgI(hipDeviceReset());
      return (1);
    }
    if (GPUFailedMsgI(hipHostMalloc(&mHostMemoryBase, mHostMemorySize))) {
      GPUError("Error allocating Page Locked Host Memory (trying %lld bytes)", (long long int)mHostMemorySize);
      GPUFailedMsgI(hipDeviceReset());
      return (1);
    }
    if (mProcessingSettings.debugLevel >= 1) {
      GPUInfo("Memory ptrs: GPU (%lld bytes): %p - Host (%lld bytes): %p", (long long int)mDeviceMemorySize, mDeviceMemoryBase, (long long int)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0, mHostMemorySize);
      if (GPUFailedMsgI(hipMemset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize))) {
        GPUError("Error during HIP memset");
        GPUFailedMsgI(hipDeviceReset());
        return (1);
      }
    }

    for (int i = 0; i < mNStreams; i++) {
      if (GPUFailedMsgI(hipStreamCreate(&mInternals->Streams[i]))) {
        GPUError("Error creating HIP Stream");
        GPUFailedMsgI(hipDeviceReset());
        return (1);
      }
    }

    void* devPtrConstantMem;
#ifndef GPUCA_NO_CONSTANT_MEMORY
    runConstantRegistrators();
    devPtrConstantMem = mDeviceConstantMemList[0];
#else
    if (GPUFailedMsgI(hipMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize))) {
      GPUError("HIP Memory Allocation Error");
      GPUFailedMsgI(hipDeviceReset());
      return (1);
    }
#endif
    mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(dummyInitKernel), dim3(mBlockCount), dim3(256), 0, 0, mDeviceMemoryBase);
    GPUInfo("HIP Initialisation successfull (Device %d: %s (Frequency %d, Cores %d), %lld / %lld bytes host / global memory, Stack frame %d, Constant memory %lld)", mDeviceId, hipDeviceProp.name, hipDeviceProp.clockRate, hipDeviceProp.multiProcessorCount, (long long int)mHostMemorySize, (long long int)mDeviceMemorySize, (int)GPUCA_GPU_STACK_SIZE, (long long int)gGPUConstantMemBufferSize);
  } else {
    GPUReconstructionHIP* master = dynamic_cast<GPUReconstructionHIP*>(mMaster);
    mDeviceId = master->mDeviceId;
    mBlockCount = master->mBlockCount;
    mWarpSize = master->mWarpSize;
    mMaxThreads = master->mMaxThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mInternals = master->mInternals;
    GPUFailedMsgI(hipSetDevice(mDeviceId));
    GPUInfo("HIP Initialized from master");
  }
  for (unsigned int i = 0; i < mEvents.size(); i++) {
    hipEvent_t* events = (hipEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
      if (GPUFailedMsgI(hipEventCreateWithFlags(&events[j], hipEventBlockingSync))) {
        GPUError("Error creating event");
        GPUFailedMsgI(hipDeviceReset());
        return 1;
      }
    }
  }

  return (0);
}

int GPUReconstructionHIP::ExitDevice_Runtime()
{
  // Uninitialize HIP
  GPUFailedMsgI(hipSetDevice(mDeviceId));
  SynchronizeGPU();
  unregisterRemainingRegisteredMemory();

  for (unsigned int i = 0; i < mEvents.size(); i++) {
    hipEvent_t* events = (hipEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
      GPUFailedMsgI(hipEventDestroy(events[j]));
    }
  }

  if (mMaster == nullptr) {
    GPUFailedMsgI(hipFree(mDeviceMemoryBase));

#ifdef GPUCA_NO_CONSTANT_MEMORY
    GPUFailedMsgI(hipFree(mDeviceConstantMem));
#endif

    for (int i = 0; i < mNStreams; i++) {
      GPUFailedMsgI(hipStreamDestroy(mInternals->Streams[i]));
    }

    GPUFailedMsgI(hipHostFree(mHostMemoryBase));
    GPUFailedMsgI(hipDeviceReset());
    GPUInfo("HIP Uninitialized");
  }
  mHostMemoryBase = nullptr;
  mDeviceMemoryBase = nullptr;

  /*if (GPUFailedMsgI(hipDeviceReset())) { // No longer doing this, another thread might use the GPU
    GPUError("Could not uninitialize GPU");
    return (1);
  }*/

  return (0);
}

size_t GPUReconstructionHIP::GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent ev, deviceEvent* evList, int nEvents)
{
  if (mProcessingSettings.debugLevel >= 3) {
    stream = -1;
  }
  if (stream == -1) {
    SynchronizeGPU();
    GPUFailedMsg(hipMemcpy(dst, src, size, toGPU ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost));
  } else {
    if (evList == nullptr) {
      nEvents = 0;
    }
    for (int k = 0; k < nEvents; k++) {
      GPUFailedMsg(hipStreamWaitEvent(mInternals->Streams[stream], ((hipEvent_t*)evList)[k], 0));
    }
    GPUFailedMsg(hipMemcpyAsync(dst, src, size, toGPU == -2 ? hipMemcpyDeviceToDevice : toGPU ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost, mInternals->Streams[stream]));
  }
  if (ev) {
    GPUFailedMsg(hipEventRecord(*(hipEvent_t*)ev, mInternals->Streams[stream == -1 ? 0 : stream]));
  }
  return size;
}

size_t GPUReconstructionHIP::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst)
{
  if (!(res->Type() & GPUMemoryResource::MEMORY_GPU)) {
    if (mProcessingSettings.debugLevel >= 4) {
      GPUInfo("Skipped transfer of non-GPU memory resource: %s", res->Name());
    }
    return 0;
  }
  if (mProcessingSettings.debugLevel >= 3 && (strcmp(res->Name(), "ErrorCodes") || mProcessingSettings.debugLevel >= 4)) {
    GPUInfo("Copying to %s: %s - %lld bytes", toGPU ? "GPU" : "Host", res->Name(), (long long int)res->Size());
  }
  return GPUMemCpy(dst, src, res->Size(), stream, toGPU, ev, evList, nEvents);
}

size_t GPUReconstructionHIP::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent ev)
{
  for (unsigned int i = 0; i < 1 + mDeviceConstantMemList.size(); i++) {
    void* basePtr = i ? mDeviceConstantMemList[i - 1] : mDeviceConstantMem;
    if (basePtr == nullptr || (i && basePtr == (void*)mDeviceConstantMem)) {
      continue;
    }
    if (stream == -1) {
      GPUFailedMsg(hipMemcpy(((char*)basePtr) + offset, src, size, hipMemcpyHostToDevice));
    } else {
      GPUFailedMsg(hipMemcpyAsync(((char*)basePtr) + offset, src, size, hipMemcpyHostToDevice, mInternals->Streams[stream]));
    }
  }
  if (ev && stream != -1) {
    GPUFailedMsg(hipEventRecord(*(hipEvent_t*)ev, mInternals->Streams[stream]));
  }
  return size;
}

void GPUReconstructionHIP::ReleaseEvent(deviceEvent ev) {}

void GPUReconstructionHIP::RecordMarker(deviceEvent ev, int stream) { GPUFailedMsg(hipEventRecord(*(hipEvent_t*)ev, mInternals->Streams[stream])); }

std::unique_ptr<GPUReconstruction::GPUThreadContext> GPUReconstructionHIP::GetThreadContext()
{
  GPUFailedMsg(hipSetDevice(mDeviceId));
  return std::unique_ptr<GPUThreadContext>(new GPUThreadContext);
}

void GPUReconstructionHIP::SynchronizeGPU() { GPUFailedMsg(hipDeviceSynchronize()); }

void GPUReconstructionHIP::SynchronizeStream(int stream) { GPUFailedMsg(hipStreamSynchronize(mInternals->Streams[stream])); }

void GPUReconstructionHIP::SynchronizeEvents(deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(hipEventSynchronize(((hipEvent_t*)evList)[i]));
  }
}

void GPUReconstructionHIP::StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(hipStreamWaitEvent(mInternals->Streams[stream], ((hipEvent_t*)evList)[i], 0));
  }
}

bool GPUReconstructionHIP::IsEventDone(deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    hipError_t retVal = hipEventSynchronize(((hipEvent_t*)evList)[i]);
    if (retVal == hipErrorNotReady) {
      return false;
    }
    GPUFailedMsg(retVal);
  }
  return (true);
}

int GPUReconstructionHIP::GPUDebug(const char* state, int stream, bool force)
{
  // Wait for HIP-Kernel to finish and check for HIP errors afterwards, in case of debugmode
  hipError_t hipErr;
  hipErr = hipGetLastError();
  if (hipErr != hipSuccess) {
    GPUError("HIP Error %s while running kernel (%s) (Stream %d)", hipGetErrorString(hipErr), state, stream);
    return (1);
  }
  if (!force && mProcessingSettings.debugLevel <= 0) {
    return (0);
  }
  if (GPUFailedMsgI(hipDeviceSynchronize())) {
    GPUError("HIP Error while synchronizing (%s) (Stream %d)", state, stream);
    return (1);
  }
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

int GPUReconstructionHIP::registerMemoryForGPU_internal(const void* ptr, size_t size)
{
  return GPUFailedMsgI(hipHostRegister((void*)ptr, size, hipHostRegisterDefault));
}

int GPUReconstructionHIP::unregisterMemoryForGPU_internal(const void* ptr)
{
  return GPUFailedMsgI(hipHostUnregister((void*)ptr));
}

void* GPUReconstructionHIP::getGPUPointer(void* ptr)
{
  void* retVal = nullptr;
  GPUFailedMsg(hipHostGetDevicePointer(&retVal, ptr, 0));
  return retVal;
}

void GPUReconstructionHIPBackend::PrintKernelOccupancies()
{

}
