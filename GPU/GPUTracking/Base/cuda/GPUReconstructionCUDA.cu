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

/// \file GPUReconstructionCUDA.cu
/// \author David Rohr

#define GPUCA_GPUCODE_HOSTONLY
#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionCUDAIncludes.h"

#include <cuda_profiler_api.h>

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "CUDAThrustHelpers.h"
#include "GPUReconstructionIncludes.h"
#include "GPUParamRTC.h"

#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
#include "utils/qGetLdBinarySymbols.h"
#ifndef __HIPCC__ // CUDA
#define PER_KERNEL_OBJECT_EXT _fatbin
#else // HIP
#define PER_KERNEL_OBJECT_EXT _hip_cxx_o
#endif
#define GPUCA_KRNL(x_class, ...) QGET_LD_BINARY_SYMBOLS(GPUCA_M_CAT3(cuda_kernel_module_fatbin_krnl_, GPUCA_M_KRNL_NAME(x_class), PER_KERNEL_OBJECT_EXT))
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

GPUReconstructionCUDABackend::GPUReconstructionCUDABackend(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionDeviceBase(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionCUDAInternals;
  }
}

GPUReconstructionCUDABackend::~GPUReconstructionCUDABackend()
{
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

int32_t GPUReconstructionCUDABackend::GPUFailedMsgAI(const int64_t error, const char* file, int32_t line)
{
  // Check for CUDA Error and in the case of an error display the corresponding error string
  if (error == cudaSuccess) {
    return (0);
  }
  GPUError("CUDA Error: %ld / %s (%s:%d)", error, cudaGetErrorString((cudaError_t)error), file, line);
  return 1;
}

void GPUReconstructionCUDABackend::GPUFailedMsgA(const int64_t error, const char* file, int32_t line)
{
  if (GPUFailedMsgAI(error, file, line)) {
    static bool runningCallbacks = false;
    if (IsInitialized() && runningCallbacks == false) {
      runningCallbacks = true;
      CheckErrorCodes(false, true);
    }
    throw std::runtime_error("CUDA Failure");
  }
}

GPUReconstructionCUDA::GPUReconstructionCUDA(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionKernels(cfg)
{
  mDeviceBackendSettings.deviceType = DeviceType::CUDA;
#ifndef __HIPCC__ // CUDA
  mRtcSrcExtension = ".cu";
  mRtcBinExtension = ".fatbin";
#else // HIP
  mRtcSrcExtension = ".hip";
  mRtcBinExtension = ".o";
#endif
}

GPUReconstructionCUDA::~GPUReconstructionCUDA()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit CUDA
}

GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCUDA(cfg); }

void GPUReconstructionCUDA::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame>* timeFrame)
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

void GPUReconstructionCUDA::UpdateAutomaticProcessingSettings()
{
  GPUCA_GPUReconstructionUpdateDefaults();
}

int32_t GPUReconstructionCUDA::InitDevice_Runtime()
{
#ifndef __HIPCC__ // CUDA
  constexpr int32_t reqVerMaj = 2;
  constexpr int32_t reqVerMin = 0;
#endif
  if (mProcessingSettings.rtc.enable && mProcessingSettings.rtc.runTest == 2) {
    genAndLoadRTC();
    exit(0);
  }

  if (mMaster == nullptr) {
    cudaDeviceProp deviceProp;
    int32_t count, bestDevice = -1;
    double bestDeviceSpeed = -1, deviceSpeed;
    if (GPUFailedMsgI(cudaGetDeviceCount(&count))) {
      GPUError("Error getting CUDA Device Count");
      return (1);
    }
    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Available CUDA devices:");
    }
    std::vector<bool> devicesOK(count, false);
    std::vector<size_t> devMemory(count, 0);
    bool contextCreated = false;
    for (int32_t i = 0; i < count; i++) {
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Examining device %d", i);
      }
      size_t free, total;
#ifndef __HIPCC__ // CUDA
      if (GPUFailedMsgI(cudaInitDevice(i, 0, 0))) {
#else // HIP
      if (GPUFailedMsgI(hipSetDevice(i))) {
#endif
        if (mProcessingSettings.debugLevel >= 4) {
          GPUWarning("Couldn't create context for device %d. Skipping it.", i);
        }
        continue;
      }
      contextCreated = true;
      if (GPUFailedMsgI(cudaMemGetInfo(&free, &total))) {
        if (mProcessingSettings.debugLevel >= 4) {
          GPUWarning("Error obtaining CUDA memory info about device %d! Skipping it.", i);
        }
        GPUFailedMsg(cudaDeviceReset());
        continue;
      }
      if (count > 1) {
        GPUFailedMsg(cudaDeviceReset());
        contextCreated = false;
      }
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Obtained current memory usage for device %d", i);
      }
      if (GPUFailedMsgI(cudaGetDeviceProperties(&deviceProp, i))) {
        continue;
      }
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Obtained device properties for device %d", i);
      }
      int32_t deviceOK = true;
      [[maybe_unused]] const char* deviceFailure = "";
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
      devMemory[i] = std::min<size_t>(free, std::max<int64_t>(0, total - REQUIRE_MEMORY_RESERVED));
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
      GPUWarning("No %sCUDA Device available, aborting CUDA Initialisation (Required mem: %ld)", count ? "appropriate " : "", (int64_t)mDeviceMemorySize);
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
        GPUFailedMsgI(cudaDeviceReset());
      }
      return (1);
    }
    mDeviceId = bestDevice;

    GPUFailedMsgI(cudaGetDeviceProperties(&deviceProp, mDeviceId));

    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Using CUDA Device %s with Properties:", deviceProp.name);
      GPUInfo("\ttotalGlobalMem = %ld", (uint64_t)deviceProp.totalGlobalMem);
      GPUInfo("\tsharedMemPerBlock = %ld", (uint64_t)deviceProp.sharedMemPerBlock);
      GPUInfo("\tregsPerBlock = %d", deviceProp.regsPerBlock);
      GPUInfo("\twarpSize = %d", deviceProp.warpSize);
      GPUInfo("\tmemPitch = %ld", (uint64_t)deviceProp.memPitch);
      GPUInfo("\tmaxThreadsPerBlock = %d", deviceProp.maxThreadsPerBlock);
      GPUInfo("\tmaxThreadsDim = %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
      GPUInfo("\tmaxGridSize = %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
      GPUInfo("\ttotalConstMem = %ld", (uint64_t)deviceProp.totalConstMem);
      GPUInfo("\tmajor = %d", deviceProp.major);
      GPUInfo("\tminor = %d", deviceProp.minor);
      GPUInfo("\tclockRate = %d", deviceProp.clockRate);
      GPUInfo("\tmemoryClockRate = %d", deviceProp.memoryClockRate);
      GPUInfo("\tmultiProcessorCount = %d", deviceProp.multiProcessorCount);
      GPUInfo("\ttextureAlignment = %ld", (uint64_t)deviceProp.textureAlignment);
      GPUInfo(" ");
    }
    if (deviceProp.warpSize != GPUCA_WARP_SIZE) {
      throw std::runtime_error("Invalid warp size on GPU");
    }
    mBlockCount = deviceProp.multiProcessorCount;
    mMaxThreads = std::max<int32_t>(mMaxThreads, deviceProp.maxThreadsPerBlock * mBlockCount);
#ifndef __HIPCC__ // CUDA
    mWarpSize = 32;
#else // HIP
    mWarpSize = 64;
#endif
    mDeviceName = deviceProp.name;
    mDeviceName += " (CUDA GPU)";

    if (deviceProp.major < 3) {
      GPUError("Unsupported CUDA Device");
      return (1);
    }

#ifdef GPUCA_USE_TEXTURES
    if (GPUCA_SLICE_DATA_MEMORY * NSLICES > (size_t)deviceProp.maxTexture1DLinear) {
      GPUError("Invalid maximum texture size of device: %ld < %ld\n", (int64_t)deviceProp.maxTexture1DLinear, (int64_t)(GPUCA_SLICE_DATA_MEMORY * NSLICES));
      return (1);
    }
#endif
#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (gGPUConstantMemBufferSize > deviceProp.totalConstMem) {
      GPUError("Insufficient constant memory available on GPU %d < %d!", (int32_t)deviceProp.totalConstMem, (int32_t)gGPUConstantMemBufferSize);
      return (1);
    }
#endif

#ifndef __HIPCC__ // CUDA
    if (contextCreated == 0 && GPUFailedMsgI(cudaInitDevice(mDeviceId, 0, 0))) {
#else // HIP
    if (contextCreated == 0 && GPUFailedMsgI(hipSetDevice(mDeviceId))) {
#endif
      GPUError("Could not set CUDA Device!");
      return (1);
    }

#ifndef __HIPCC__ // CUDA
    if (GPUFailedMsgI(cudaDeviceSetLimit(cudaLimitStackSize, GPUCA_GPU_STACK_SIZE))) {
      GPUError("Error setting CUDA stack size");
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
    if (GPUFailedMsgI(cudaDeviceSetLimit(cudaLimitMallocHeapSize, mProcessingSettings.deterministicGPUReconstruction ? std::max<size_t>(1024 * 1024 * 1024, GPUCA_GPU_HEAP_SIZE) : GPUCA_GPU_HEAP_SIZE))) {
      GPUError("Error setting CUDA stack size");
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
#else // HIP
    if (GPUFailedMsgI(hipSetDeviceFlags(hipDeviceScheduleBlockingSync))) {
      GPUError("Could not set HIP Device flags!");
      return (1);
    }
#endif

    if (mDeviceMemorySize == 1 || mDeviceMemorySize == 2) {
      mDeviceMemorySize = std::max<int64_t>(0, devMemory[mDeviceId] - REQUIRE_FREE_MEMORY_RESERVED_PER_SM * deviceProp.multiProcessorCount); // Take all GPU memory but some reserve
      if (mDeviceMemorySize >= RESERVE_EXTRA_MEM_THRESHOLD) {
        mDeviceMemorySize -= RESERVE_EXTRA_MEM_OFFSET;
      }
    }
    if (mDeviceMemorySize == 2) {
      mDeviceMemorySize = mDeviceMemorySize * 2 / 3; // Leave 1/3 of GPU memory for event display
    }

    if (mProcessingSettings.debugLevel >= 3) {
      GPUInfo("Allocating memory on GPU");
    }
    if (mDeviceMemorySize > deviceProp.totalGlobalMem || GPUFailedMsgI(cudaMalloc(&mDeviceMemoryBase, mDeviceMemorySize))) {
      size_t free, total;
      GPUFailedMsg(cudaMemGetInfo(&free, &total));
      GPUError("CUDA Memory Allocation Error (trying %ld bytes, %ld available on GPU, %ld free)", (int64_t)mDeviceMemorySize, (int64_t)deviceProp.totalGlobalMem, (int64_t)free);
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
    if (mProcessingSettings.debugLevel >= 3) {
      GPUInfo("Allocating memory on Host");
    }
    if (GPUFailedMsgI(cudaMallocHost(&mHostMemoryBase, mHostMemorySize))) {
      GPUError("Error allocating Page Locked Host Memory (trying %ld bytes)", (int64_t)mHostMemorySize);
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
    if (mProcessingSettings.debugLevel >= 1) {
      GPUInfo("Memory ptrs: GPU (%ld bytes): %p - Host (%ld bytes): %p", (int64_t)mDeviceMemorySize, mDeviceMemoryBase, (int64_t)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
      if (GPUFailedMsgI(cudaMemset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize))) {
        GPUError("Error during CUDA memset");
        GPUFailedMsgI(cudaDeviceReset());
        return (1);
      }
    }

    for (int32_t i = 0; i < mNStreams; i++) {
      if (GPUFailedMsgI(cudaStreamCreateWithFlags(&mInternals->Streams[i], cudaStreamNonBlocking))) {
        GPUError("Error creating CUDA Stream");
        GPUFailedMsgI(cudaDeviceReset());
        return (1);
      }
    }

#ifndef __HIPCC__ // CUDA
    dummyInitKernel<<<mBlockCount, 256>>>(mDeviceMemoryBase);
#else // HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dummyInitKernel), dim3(mBlockCount), dim3(256), 0, 0, mDeviceMemoryBase);
#endif

#ifndef GPUCA_ALIROOT_LIB
    if (mProcessingSettings.rtc.enable) {
      genAndLoadRTC();
    }
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
    else {
#define GPUCA_KRNL(x_class, ...)                                        \
  mInternals->kernelModules.emplace_back(std::make_unique<CUmodule>()); \
  GPUFailedMsg(cuModuleLoadData(mInternals->kernelModules.back().get(), GPUCA_M_CAT3(_binary_cuda_kernel_module_fatbin_krnl_, GPUCA_M_KRNL_NAME(x_class), GPUCA_M_CAT(PER_KERNEL_OBJECT_EXT, _start))));
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
    for (uint32_t i = 0; i < mInternals->kernelModules.size(); i++) {
#ifndef __HIPCC__
      CUdeviceptr tmp; // CUDA has a custom type, that initializes to zero and cannot be initialized with nullptr
#else
      CUdeviceptr tmp = nullptr; // HIP just uses void*
#endif
      size_t tmpSize = 0;
      GPUFailedMsg(cuModuleGetGlobal(&tmp, &tmpSize, *mInternals->kernelModules[i], "gGPUConstantMemBuffer"));
      mDeviceConstantMemList.emplace_back((void*)tmp);
    }
#else
    GPUFailedMsg(cudaMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize));
#endif
    mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;

    GPUInfo("CUDA Initialisation successfull (Device %d: %s (Frequency %d, Cores %d), %ld / %ld bytes host / global memory, Stack frame %d, Constant memory %ld)", mDeviceId, deviceProp.name, deviceProp.clockRate, deviceProp.multiProcessorCount, (int64_t)mHostMemorySize, (int64_t)mDeviceMemorySize, (int32_t)GPUCA_GPU_STACK_SIZE, (int64_t)gGPUConstantMemBufferSize);
  } else {
    GPUReconstructionCUDA* master = dynamic_cast<GPUReconstructionCUDA*>(mMaster);
    mDeviceId = master->mDeviceId;
    mBlockCount = master->mBlockCount;
    mWarpSize = master->mWarpSize;
    mMaxThreads = master->mMaxThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mDeviceConstantMemList.resize(master->mDeviceConstantMemList.size());
    std::copy(master->mDeviceConstantMemList.begin(), master->mDeviceConstantMemList.end(), mDeviceConstantMemList.begin());
    mInternals = master->mInternals;
    GPUFailedMsg(cudaSetDevice(mDeviceId));

    GPUInfo("CUDA Initialized from master");
  }

  for (uint32_t i = 0; i < mEvents.size(); i++) {
    cudaEvent_t* events = (cudaEvent_t*)mEvents[i].data();
    for (uint32_t j = 0; j < mEvents[i].size(); j++) {
#ifndef __HIPCC__ // CUDA
      if (GPUFailedMsgI(cudaEventCreate(&events[j]))) {
#else
      if (GPUFailedMsgI(hipEventCreateWithFlags(&events[j], hipEventBlockingSync))) {
#endif
        GPUError("Error creating event");
        GPUFailedMsgI(cudaDeviceReset());
        return 1;
      }
    }
  }

  return (0);
}

void GPUReconstructionCUDA::genAndLoadRTC()
{
  std::string filename = "";
  uint32_t nCompile = 0;
  if (genRTC(filename, nCompile)) {
    throw std::runtime_error("Runtime compilation failed");
  }
  for (uint32_t i = 0; i < nCompile; i++) {
    if (mProcessingSettings.rtc.runTest != 2) {
      mInternals->kernelModules.emplace_back(std::make_unique<CUmodule>());
      GPUFailedMsg(cuModuleLoad(mInternals->kernelModules.back().get(), (filename + "_" + std::to_string(i) + mRtcBinExtension).c_str()));
    }
    remove((filename + "_" + std::to_string(i) + mRtcSrcExtension).c_str());
    remove((filename + "_" + std::to_string(i) + mRtcBinExtension).c_str());
  }
  if (mProcessingSettings.rtc.runTest == 2) {
    return;
  }
  loadKernelModules(mProcessingSettings.rtc.compilePerKernel);
}

int32_t GPUReconstructionCUDA::ExitDevice_Runtime()
{
  // Uninitialize CUDA
  GPUFailedMsg(cudaSetDevice(mDeviceId));
  SynchronizeGPU();
  unregisterRemainingRegisteredMemory();

  for (uint32_t i = 0; i < mEvents.size(); i++) {
    cudaEvent_t* events = (cudaEvent_t*)mEvents[i].data();
    for (uint32_t j = 0; j < mEvents[i].size(); j++) {
      GPUFailedMsgI(cudaEventDestroy(events[j]));
    }
  }

  if (mMaster == nullptr) {
    GPUFailedMsgI(cudaFree(mDeviceMemoryBase));
#ifdef GPUCA_NO_CONSTANT_MEMORY
    GPUFailedMsgI(cudaFree(mDeviceConstantMem));
#endif

    for (int32_t i = 0; i < mNStreams; i++) {
      GPUFailedMsgI(cudaStreamDestroy(mInternals->Streams[i]));
    }

    GPUFailedMsgI(cudaFreeHost(mHostMemoryBase));
    for (uint32_t i = 0; i < mInternals->kernelModules.size(); i++) {
      GPUFailedMsg(cuModuleUnload(*mInternals->kernelModules[i]));
    }

    GPUFailedMsgI(cudaDeviceReset());
    GPUInfo("CUDA Uninitialized");
  }
  mDeviceMemoryBase = nullptr;
  mHostMemoryBase = nullptr;

  return (0);
}

size_t GPUReconstructionCUDA::GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  if (mProcessingSettings.debugLevel >= 3) {
    stream = -1;
  }
  if (stream == -1) {
    SynchronizeGPU();
    GPUFailedMsg(cudaMemcpy(dst, src, size, toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));
  } else {
    if (evList == nullptr) {
      nEvents = 0;
    }
    for (int32_t k = 0; k < nEvents; k++) {
      GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[stream], evList[k].get<cudaEvent_t>(), 0));
    }
    GPUFailedMsg(cudaMemcpyAsync(dst, src, size, toGPU == -2 ? cudaMemcpyDeviceToDevice : toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, mInternals->Streams[stream]));
  }
  if (ev) {
    GPUFailedMsg(cudaEventRecord(ev->get<cudaEvent_t>(), mInternals->Streams[stream == -1 ? 0 : stream]));
  }
  if (mProcessingSettings.serializeGPU & 2) {
    GPUDebug(("GPUMemCpy " + std::to_string(toGPU)).c_str(), stream, true);
  }
  return size;
}

size_t GPUReconstructionCUDA::WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream, deviceEvent* ev)
{
  for (uint32_t i = 0; i < 1 + mDeviceConstantMemList.size(); i++) {
    void* basePtr = i ? mDeviceConstantMemList[i - 1] : mDeviceConstantMem;
    if (basePtr == nullptr || (i && basePtr == (void*)mDeviceConstantMem)) {
      continue;
    }
    if (stream == -1) {
      GPUFailedMsg(cudaMemcpy(((char*)basePtr) + offset, src, size, cudaMemcpyHostToDevice));
    } else {
      GPUFailedMsg(cudaMemcpyAsync(((char*)basePtr) + offset, src, size, cudaMemcpyHostToDevice, mInternals->Streams[stream]));
    }
  }
  if (ev && stream != -1) {
    GPUFailedMsg(cudaEventRecord(ev->get<cudaEvent_t>(), mInternals->Streams[stream]));
  }
  if (mProcessingSettings.serializeGPU & 2) {
    GPUDebug("WriteToConstantMemory", stream, true);
  }
  return size;
}

void GPUReconstructionCUDA::ReleaseEvent(deviceEvent ev) {}
void GPUReconstructionCUDA::RecordMarker(deviceEvent ev, int32_t stream) { GPUFailedMsg(cudaEventRecord(ev.get<cudaEvent_t>(), mInternals->Streams[stream])); }

std::unique_ptr<GPUReconstruction::GPUThreadContext> GPUReconstructionCUDA::GetThreadContext()
{
  GPUFailedMsg(cudaSetDevice(mDeviceId));
  return std::unique_ptr<GPUThreadContext>(new GPUThreadContext);
}

void GPUReconstructionCUDA::SynchronizeGPU() { GPUFailedMsg(cudaDeviceSynchronize()); }
void GPUReconstructionCUDA::SynchronizeStream(int32_t stream) { GPUFailedMsg(cudaStreamSynchronize(mInternals->Streams[stream])); }

void GPUReconstructionCUDA::SynchronizeEvents(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    GPUFailedMsg(cudaEventSynchronize(evList[i].get<cudaEvent_t>()));
  }
}

void GPUReconstructionCUDA::StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[stream], evList[i].get<cudaEvent_t>(), 0));
  }
}

bool GPUReconstructionCUDA::IsEventDone(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    cudaError_t retVal = cudaEventSynchronize(evList[i].get<cudaEvent_t>());
    if (retVal == cudaErrorNotReady) {
      return false;
    }
    GPUFailedMsg(retVal);
  }
  return (true);
}

int32_t GPUReconstructionCUDA::GPUDebug(const char* state, int32_t stream, bool force)
{
  // Wait for CUDA-Kernel to finish and check for CUDA errors afterwards, in case of debugmode
  cudaError cuErr;
  cuErr = cudaGetLastError();
  if (cuErr != cudaSuccess) {
    GPUError("CUDA Error %s while running (%s) (Stream %d)", cudaGetErrorString(cuErr), state, stream);
    return (1);
  }
  if (!force && mProcessingSettings.debugLevel <= 0) {
    return (0);
  }
  if (GPUFailedMsgI(stream == -1 ? cudaDeviceSynchronize() : cudaStreamSynchronize(mInternals->Streams[stream]))) {
    GPUError("CUDA Error while synchronizing (%s) (Stream %d)", state, stream);
    return (1);
  }
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

int32_t GPUReconstructionCUDA::registerMemoryForGPU_internal(const void* ptr, size_t size)
{
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("Registering %zu bytes of memory for GPU", size);
  }
  return GPUFailedMsgI(cudaHostRegister((void*)ptr, size, cudaHostRegisterDefault));
}

int32_t GPUReconstructionCUDA::unregisterMemoryForGPU_internal(const void* ptr)
{
  return GPUFailedMsgI(cudaHostUnregister((void*)ptr));
}

void GPUReconstructionCUDABackend::PrintKernelOccupancies()
{
  int32_t maxBlocks = 0, threads = 0, suggestedBlocks = 0, nRegs = 0, sMem = 0;
  GPUFailedMsg(cudaSetDevice(mDeviceId));
  for (uint32_t i = 0; i < mInternals->kernelFunctions.size(); i++) {
    GPUFailedMsg(cuOccupancyMaxPotentialBlockSize(&suggestedBlocks, &threads, *mInternals->kernelFunctions[i], 0, 0, 0));
    GPUFailedMsg(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, *mInternals->kernelFunctions[i], threads, 0));
    GPUFailedMsg(cuFuncGetAttribute(&nRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, *mInternals->kernelFunctions[i]));
    GPUFailedMsg(cuFuncGetAttribute(&sMem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *mInternals->kernelFunctions[i]));
    GPUInfo("Kernel: %50s Block size: %4d, Maximum active blocks: %3d, Suggested blocks: %3d, Regs: %3d, smem: %3d", mInternals->kernelNames[i].c_str(), threads, maxBlocks, suggestedBlocks, nRegs, sMem);
  }
}

void GPUReconstructionCUDA::loadKernelModules(bool perKernel, bool perSingleMulti)
{
  uint32_t j = 0;
#define GPUCA_KRNL(...)                          \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, __VA_ARGS__) \
  j += !perSingleMulti;
#define GPUCA_KRNL_LOAD_single(x_class, ...)                                                                                                                                               \
  getRTCkernelNum<false, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->kernelFunctions.size());                                                                                              \
  mInternals->kernelFunctions.emplace_back(new CUfunction);                                                                                                                                \
  mInternals->kernelNames.emplace_back(GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))));                                                                                       \
  if (mProcessingSettings.debugLevel >= 3) {                                                                                                                                               \
    GPUInfo("Loading kernel %s (j = %u)", GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), j);                                                                                 \
  }                                                                                                                                                                                        \
  GPUFailedMsg(cuModuleGetFunction(mInternals->kernelFunctions.back().get(), *mInternals->kernelModules[perKernel ? j : 0], GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))))); \
  j += perSingleMulti;
#define GPUCA_KRNL_LOAD_multi(x_class, ...)                                                                                                                                                         \
  getRTCkernelNum<true, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->kernelFunctions.size());                                                                                                        \
  mInternals->kernelFunctions.emplace_back(new CUfunction);                                                                                                                                         \
  mInternals->kernelNames.emplace_back(GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)));                                                                                       \
  if (mProcessingSettings.debugLevel >= 3) {                                                                                                                                                        \
    GPUInfo("Loading kernel %s (j = %u)", GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)), j);                                                                                 \
  }                                                                                                                                                                                                 \
  GPUFailedMsg(cuModuleGetFunction(mInternals->kernelFunctions.back().get(), *mInternals->kernelModules[perKernel ? j : 0], GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)))); \
  j += perSingleMulti;
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

  if (j != mInternals->kernelModules.size()) {
    GPUFatal("Did not load all kernels (%u < %u)", j, (uint32_t)mInternals->kernelModules.size());
  }
}

#ifndef __HIPCC__ // CUDA
int32_t GPUReconstructionCUDA::PrepareTextures()
{
#ifdef GPUCA_USE_TEXTURES
  cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<cahit2>();
  size_t offset;
  GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu2, mProcessorsShadow->tpcTrackers[0].Data().Memory(), &channelDescu2, NSLICES * GPUCA_SLICE_DATA_MEMORY));
  cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<calink>();
  GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu, mProcessorsShadow->tpcTrackers[0].Data().Memory(), &channelDescu, NSLICES * GPUCA_SLICE_DATA_MEMORY));
#endif
  return (0);
}

void GPUReconstructionCUDA::startGPUProfiling()
{
  GPUFailedMsg(cudaProfilerStart());
}

void GPUReconstructionCUDA::endGPUProfiling()
{
  GPUFailedMsg(cudaProfilerStop());
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
template class GPUReconstructionKernels<GPUReconstructionCUDABackend>;
}
