#include "hip/hip_runtime.h"
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

/// \file GPUReconstructionHIP.cu
/// \author David Rohr

#define GPUCA_GPUCODE_HOSTONLY
#include "GPUReconstructionHIPDef.h"
#include "GPUReconstructionHIPIncludes.h"

#include <hip/hip_runtime_api.h>

#include "GPUReconstructionHIP.h"
#include "GPUReconstructionHIPInternals.h"
#include "HIPThrustHelpers.h"
#include "GPUReconstructionIncludes.h"
#include "GPUParamRTC.h"

#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
#include "utils/qGetLdBinarySymbols.h"
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) QGET_LD_BINARY_SYMBOLS(GPUCA_M_CAT3(hip_kernel_module_fatbin_krnl_, GPUCA_M_KRNL_NAME(x_class), _fatbin))
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
#endif

static constexpr size_t REQUIRE_MIN_MEMORY = 1024L * 1024 * 1024;
static constexpr size_t REQUIRE_MEMORY_RESERVED = 512L * 1024 * 1024;
static constexpr size_t REQUIRE_FREE_MEMORY_RESERVED_PER_SM = 40L * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_THRESHOLD = 10L * 1024 * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_OFFSET = 1L * 512 * 1024 * 1024;

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
    for (unsigned int i = 0; i < mInternals->kernelModules.size(); i++) {
      GPUFailedMsg(hipModuleUnload(*mInternals->kernelModules[i]));
    }
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
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit HIP
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
#ifndef __HIPCC__ // HIP
  constexpr int reqVerMaj = 2;
  constexpr int reqVerMin = 0;
#endif
  if (mMaster == nullptr) {
    hipDeviceProp_t deviceProp;
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
#ifndef __HIPCC__ // HIP
      if (GPUFailedMsgI(hipInitDevice(i, 0, 0))) {
#else // HIP
      if (GPUFailedMsgI(hipSetDevice(i))) {
#endif
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
      if (GPUFailedMsgI(hipGetDeviceProperties(&deviceProp, i))) {
        continue;
      }
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Obtained device properties for device %d", i);
      }
      int deviceOK = true;
      const char* deviceFailure = "";
#ifndef __HIPCC__
      if (deviceProp.major < reqVerMaj || (deviceProp.major == reqVerMaj && deviceProp.minor < reqVerMin)) {
        deviceOK = false;
        deviceFailure = "Too low device revision";
      }
#endif
      if (free < std::max<size_t>(mDeviceMemorySize, REQUIRE_MIN_MEMORY)) {
        deviceOK = false;
        deviceFailure = "Insufficient GPU memory";
      }

      deviceSpeed = (double)deviceProp.multiProcessorCount * (double)deviceProp.clockRate * (double)deviceProp.warpSize * (double)free * (double)deviceProp.major * (double)deviceProp.major;
      if (mProcessingSettings.debugLevel >= 2) {
        GPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem Avail %lu / %lu)%s %s", deviceOK ? " " : "[", i, deviceProp.name, deviceProp.major, deviceProp.minor, free, (size_t)deviceProp.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
      }
      if (!deviceOK) {
        continue;
      }
      devicesOK[i] = true;
      devMemory[i] = std::min<size_t>(free, std::max<long int>(0, total - REQUIRE_MEMORY_RESERVED));
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
#ifndef __HIPCC__
      GPUImportant("Requiring Revision %d.%d, Mem: %lu", reqVerMaj, reqVerMin, std::max<size_t>(mDeviceMemorySize, REQUIRE_MIN_MEMORY));
#endif
      noDevice = true;
    } else if (mProcessingSettings.deviceNum > -1) {
      if (mProcessingSettings.deviceNum >= (signed)count) {
        GPUError("Requested device ID %d does not exist", mProcessingSettings.deviceNum);
        noDevice = true;
      } else if (!devicesOK[mProcessingSettings.deviceNum]) {
        GPUError("Unsupported device requested (%d)", mProcessingSettings.deviceNum);
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

    GPUFailedMsgI(hipGetDeviceProperties(&deviceProp, mDeviceId));

    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Using HIP Device %s with Properties:", deviceProp.name);
      GPUInfo("\ttotalGlobalMem = %lld", (unsigned long long int)deviceProp.totalGlobalMem);
      GPUInfo("\tsharedMemPerBlock = %lld", (unsigned long long int)deviceProp.sharedMemPerBlock);
      GPUInfo("\tregsPerBlock = %d", deviceProp.regsPerBlock);
      GPUInfo("\twarpSize = %d", deviceProp.warpSize);
      GPUInfo("\tmemPitch = %lld", (unsigned long long int)deviceProp.memPitch);
      GPUInfo("\tmaxThreadsPerBlock = %d", deviceProp.maxThreadsPerBlock);
      GPUInfo("\tmaxThreadsDim = %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
      GPUInfo("\tmaxGridSize = %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
      GPUInfo("\ttotalConstMem = %lld", (unsigned long long int)deviceProp.totalConstMem);
      GPUInfo("\tmajor = %d", deviceProp.major);
      GPUInfo("\tminor = %d", deviceProp.minor);
      GPUInfo("\tclockRate = %d", deviceProp.clockRate);
      GPUInfo("\tmemoryClockRate = %d", deviceProp.memoryClockRate);
      GPUInfo("\tmultiProcessorCount = %d", deviceProp.multiProcessorCount);
      GPUInfo("\ttextureAlignment = %lld", (unsigned long long int)deviceProp.textureAlignment);
      GPUInfo(" ");
    }
    if (deviceProp.warpSize != GPUCA_WARP_SIZE) {
      throw std::runtime_error("Invalid warp size on GPU");
    }
    mBlockCount = deviceProp.multiProcessorCount;
    mMaxThreads = std::max<int>(mMaxThreads, deviceProp.maxThreadsPerBlock * mBlockCount);
#ifndef __HIPCC__ // HIP
    mWarpSize = 32;
#else // HIP
    mWarpSize = 64;
#endif
    mDeviceName = deviceProp.name;
    mDeviceName += " (HIP GPU)";

    if (deviceProp.major < 3) {
      GPUError("Unsupported HIP Device");
      return (1);
    }

#ifdef GPUCA_USE_TEXTURES
    if (GPUCA_SLICE_DATA_MEMORY * NSLICES > (size_t)deviceProp.maxTexture1DLinear) {
      GPUError("Invalid maximum texture size of device: %lld < %lld\n", (long long int)deviceProp.maxTexture1DLinear, (long long int)(GPUCA_SLICE_DATA_MEMORY * NSLICES));
      return (1);
    }
#endif
#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (gGPUConstantMemBufferSize > deviceProp.totalConstMem) {
      GPUError("Insufficient constant memory available on GPU %d < %d!", (int)deviceProp.totalConstMem, (int)gGPUConstantMemBufferSize);
      return (1);
    }
#endif

#ifndef __HIPCC__ // HIP
    if (contextCreated == 0 && GPUFailedMsgI(hipInitDevice(mDeviceId, 0, 0))) {
#else // HIP
    if (contextCreated == 0 && GPUFailedMsgI(hipSetDevice(mDeviceId))) {
#endif
      GPUError("Could not set HIP Device!");
      return (1);
    }

#ifndef __HIPCC__ // HIP
    if (GPUFailedMsgI(hipDeviceSetLimit(hipLimitStackSize, GPUCA_GPU_STACK_SIZE))) {
      GPUError("Error setting HIP stack size");
      GPUFailedMsgI(hipDeviceReset());
      return (1);
    }
    if (GPUFailedMsgI(hipDeviceSetLimit(hipLimitMallocHeapSize, mProcessingSettings.deterministicGPUReconstruction ? std::max<size_t>(1024 * 1024 * 1024, GPUCA_GPU_HEAP_SIZE) : GPUCA_GPU_HEAP_SIZE))) {
      GPUError("Error setting HIP stack size");
      GPUFailedMsgI(hipDeviceReset());
      return (1);
    }
#else // HIP
    if (GPUFailedMsgI(hipSetDeviceFlags(hipDeviceScheduleBlockingSync))) {
      GPUError("Could not set HIP Device!");
      return (1);
    }
#endif

    if (mDeviceMemorySize == 1 || mDeviceMemorySize == 2) {
      mDeviceMemorySize = std::max<long int>(0, devMemory[mDeviceId] - REQUIRE_FREE_MEMORY_RESERVED_PER_SM * deviceProp.multiProcessorCount); // Take all GPU memory but some reserve
      if (mDeviceMemorySize >= RESERVE_EXTRA_MEM_THRESHOLD) {
        mDeviceMemorySize -= RESERVE_EXTRA_MEM_OFFSET;
      }
    }
    if (mDeviceMemorySize == 2) {
      mDeviceMemorySize = mDeviceMemorySize * 2 / 3; // Leave 1/3 of GPU memory for event display
    }

    if (mDeviceMemorySize > deviceProp.totalGlobalMem || GPUFailedMsgI(hipMalloc(&mDeviceMemoryBase, mDeviceMemorySize))) {
      size_t free, total;
      GPUFailedMsg(hipMemGetInfo(&free, &total));
      GPUError("HIP Memory Allocation Error (trying %lld bytes, %lld available on GPU, %lld free)", (long long int)mDeviceMemorySize, (long long int)deviceProp.totalGlobalMem, (long long int)free);
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
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
      if (GPUFailedMsgI(hipMemset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize))) {
        GPUError("Error during HIP memset");
        GPUFailedMsgI(hipDeviceReset());
        return (1);
      }
    }

    for (int i = 0; i < mNStreams; i++) {
      if (GPUFailedMsgI(hipStreamCreateWithFlags(&mInternals->Streams[i], hipStreamNonBlocking))) {
        GPUError("Error creating HIP Stream");
        GPUFailedMsgI(hipDeviceReset());
        return (1);
      }
    }

#ifndef __HIPCC__ // HIP
    dummyInitKernel<<<mBlockCount, 256>>>(mDeviceMemoryBase);
#else // HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dummyInitKernel), dim3(mBlockCount), dim3(256), 0, 0, mDeviceMemoryBase);
#endif

#ifndef GPUCA_ALIROOT_LIB
    if (mProcessingSettings.rtc.enable) {
      std::string filename = "";
      unsigned int nCompile = 0;
      if (genRTC(filename, nCompile)) {
        throw std::runtime_error("Runtime compilation failed");
      }
      for (unsigned int i = 0; i < nCompile; i++) {
        mInternals->kernelModules.emplace_back(std::make_unique<hipModule_t>());
        GPUFailedMsg(hipModuleLoad(mInternals->kernelModules.back().get(), (filename + "_" + std::to_string(i) + ".cubin").c_str()));
        remove((filename + "_" + std::to_string(i) + ".cu").c_str());
        remove((filename + "_" + std::to_string(i) + ".cubin").c_str());
      }
      loadKernelModules(mProcessingSettings.rtc.compilePerKernel);
    }
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
    else {
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward)          \
  mInternals->kernelModules.emplace_back(std::make_unique<hipModule_t>()); \
  GPUFailedMsg(hipModuleLoadData(mInternals->kernelModules.back().get(), GPUCA_M_CAT3(_binary_hip_kernel_module_fatbin_krnl_, GPUCA_M_KRNL_NAME(x_class), _fatbin_start)));
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
      loadKernelModules(true, false);
    }
#endif
#endif
    void* devPtrConstantMem = nullptr;
#ifndef GPUCA_NO_CONSTANT_MEMORY
    runConstantRegistrators();
    devPtrConstantMem = mDeviceConstantMemList[0];
    for (unsigned int i = 0; i < mInternals->kernelModules.size(); i++) {
      hipDeviceptr_t tmp;
      GPUFailedMsg(hipModuleGetGlobal(&tmp, nullptr, *mInternals->kernelModules[i], "gGPUConstantMemBuffer"));
      mDeviceConstantMemList.emplace_back((void*)tmp);
    }
#else
    GPUFailedMsg(hipMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize));
#endif
    mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;

    GPUInfo("HIP Initialisation successfull (Device %d: %s (Frequency %d, Cores %d), %lld / %lld bytes host / global memory, Stack frame %d, Constant memory %lld)", mDeviceId, deviceProp.name, deviceProp.clockRate, deviceProp.multiProcessorCount, (long long int)mHostMemorySize, (long long int)mDeviceMemorySize, (int)GPUCA_GPU_STACK_SIZE, (long long int)gGPUConstantMemBufferSize);
  } else {
    GPUReconstructionHIP* master = dynamic_cast<GPUReconstructionHIP*>(mMaster);
    mDeviceId = master->mDeviceId;
    mBlockCount = master->mBlockCount;
    mWarpSize = master->mWarpSize;
    mMaxThreads = master->mMaxThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mDeviceConstantMemList.resize(master->mDeviceConstantMemList.size());
    std::copy(master->mDeviceConstantMemList.begin(), master->mDeviceConstantMemList.end(), mDeviceConstantMemList.begin());
    mInternals = master->mInternals;
    GPUFailedMsg(hipSetDevice(mDeviceId));

    GPUInfo("HIP Initialized from master");
  }

  for (unsigned int i = 0; i < mEvents.size(); i++) {
    hipEvent_t* events = (hipEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
#ifndef __HIPCC__ // HIP
      if (GPUFailedMsgI(hipEventCreate(&events[j]))) {
#else
      if (GPUFailedMsgI(hipEventCreateWithFlags(&events[j], hipEventBlockingSync))) {
#endif
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
  GPUFailedMsg(hipSetDevice(mDeviceId));
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
  mDeviceMemoryBase = nullptr;
  mHostMemoryBase = nullptr;

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
    GPUFailedMsg(hipMemcpyAsync(dst, src, size, toGPU == -2 ? hipMemcpyDeviceToDevice : toGPU ? hipMemcpyHostToDevice
                                                                                              : hipMemcpyDeviceToHost,
                                mInternals->Streams[stream]));
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
  hipError_t cuErr;
  cuErr = hipGetLastError();
  if (cuErr != hipSuccess) {
    GPUError("HIP Error %s while running kernel (%s) (Stream %d)", hipGetErrorString(cuErr), state, stream);
    return (1);
  }
  if (!force && mProcessingSettings.debugLevel <= 0) {
    return (0);
  }
  if (GPUFailedMsgI(stream == -1 ? hipDeviceSynchronize() : hipStreamSynchronize(mInternals->Streams[stream]))) {
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

void GPUReconstructionHIPBackend::PrintKernelOccupancies()
{
  int maxBlocks = 0, threads = 0, suggestedBlocks = 0, nRegs = 0, sMem = 0;
  GPUFailedMsg(hipSetDevice(mDeviceId));
  for (unsigned int i = 0; i < mInternals->kernelFunctions.size(); i++) {
    GPUFailedMsg(hipModuleOccupancyMaxPotentialBlockSize(&suggestedBlocks, &threads, *mInternals->kernelFunctions[i], 0, 0));
    GPUFailedMsg(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, *mInternals->kernelFunctions[i], threads, 0));
    GPUFailedMsg(hipFuncGetAttribute(&nRegs, HIP_FUNC_ATTRIBUTE_NUM_REGS, *mInternals->kernelFunctions[i]));
    GPUFailedMsg(hipFuncGetAttribute(&sMem, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *mInternals->kernelFunctions[i]));
    GPUInfo("Kernel: %50s Block size: %4d, Maximum active blocks: %3d, Suggested blocks: %3d, Regs: %3d, smem: %3d", mInternals->kernelNames[i].c_str(), threads, maxBlocks, suggestedBlocks, nRegs, sMem);
  }
}

int GPUReconstructionHIP::loadKernelModules(bool perKernel, bool perSingleMulti)
{
  int j = 0;
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward)                  \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward) \
  j += !perSingleMulti;
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward)                        \
  getRTCkernelNum<false, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->kernelFunctions.size());        \
  mInternals->kernelFunctions.emplace_back(new hipFunction_t);                                       \
  mInternals->kernelNames.emplace_back(GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class)))); \
  GPUFailedMsg(hipModuleGetFunction(mInternals->kernelFunctions.back().get(), *mInternals->kernelModules[perKernel ? (j += perSingleMulti) : 0], GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class)))));
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward)                                  \
  getRTCkernelNum<true, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->kernelFunctions.size());                  \
  mInternals->kernelFunctions.emplace_back(new hipFunction_t);                                                \
  mInternals->kernelNames.emplace_back(GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi))); \
  GPUFailedMsg(hipModuleGetFunction(mInternals->kernelFunctions.back().get(), *mInternals->kernelModules[perKernel ? (j += perSingleMulti) : 0], GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi))));
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

  for (unsigned int i = 0; i < mInternals->kernelNames.size(); i++) {
    if (mProcessingSettings.debugLevel >= 3) {
      GPUInfo("Loaded module for kernel %s", mInternals->kernelNames[i].c_str());
    }
  }
  return 0;
}

#ifndef __HIPCC__ // HIP
int GPUReconstructionHIP::PrepareTextures()
{
#ifdef GPUCA_USE_TEXTURES
  hipChannelFormatDesc channelDescu2 = hipCreateChannelDesc<cahit2>();
  size_t offset;
  GPUFailedMsg(hipBindTexture(&offset, &gAliTexRefu2, mProcessorsShadow->tpcTrackers[0].Data().Memory(), &channelDescu2, NSLICES * GPUCA_SLICE_DATA_MEMORY));
  hipChannelFormatDesc channelDescu = hipCreateChannelDesc<calink>();
  GPUFailedMsg(hipBindTexture(&offset, &gAliTexRefu, mProcessorsShadow->tpcTrackers[0].Data().Memory(), &channelDescu, NSLICES * GPUCA_SLICE_DATA_MEMORY));
#endif
  return (0);
}

void GPUReconstructionHIP::startGPUProfiling()
{
  GPUFailedMsg(hipProfilerStart());
}

void GPUReconstructionHIP::endGPUProfiling()
{
  GPUFailedMsg(hipProfilerStop());
}
#else  // HIP
void* GPUReconstructionHIP::getGPUPointer(void* ptr)
{
  void* retVal = nullptr;
  GPUFailedMsg(hipHostGetDevicePointer(&retVal, ptr, 0));
  return retVal;
}
#endif // __HIPCC__

namespace GPUCA_NAMESPACE::gpu
{
template class GPUReconstructionKernels<GPUReconstructionHIPBackend>;
}
