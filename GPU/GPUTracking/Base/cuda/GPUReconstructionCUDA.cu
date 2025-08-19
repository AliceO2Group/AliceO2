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

#include "GPUReconstructionCUDAIncludesSystem.h"
#include "GPUReconstructionCUDADef.h"
#include <cuda_profiler_api.h>

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "GPUReconstructionIncludes.h"
#include "GPUParamRTC.h"
#include "GPUReconstructionCUDAHelpers.inc"
#include "GPUDefParametersLoad.inc"
#include "GPUReconstructionKernelIncludes.h"
#include "GPUConstantMem.h"

#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
#include "utils/qGetLdBinarySymbols.h"
#ifndef __HIPCC__ // CUDA
#define PER_KERNEL_OBJECT_EXT _fatbin
#else // HIP
#define PER_KERNEL_OBJECT_EXT _hip_o
#endif
#ifdef GPUCA_RTC_NO_COMPILED_KERNELS
#define GPUCA_KRNL(x_class, ...) static void* GPUCA_M_CAT3(_binary_cuda_kernel_module_fatbin_krnl_, GPUCA_M_KRNL_NAME(x_class), GPUCA_M_CAT(PER_KERNEL_OBJECT_EXT, _start)) = nullptr;
#else
#define GPUCA_KRNL(x_class, ...) QGET_LD_BINARY_SYMBOLS(GPUCA_M_CAT3(cuda_kernel_module_fatbin_krnl_, GPUCA_M_KRNL_NAME(x_class), PER_KERNEL_OBJECT_EXT))
#endif
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
#endif

#ifdef GPUCA_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

static constexpr size_t REQUIRE_MIN_MEMORY = 1024L * 1024 * 1024;
static constexpr size_t REQUIRE_MEMORY_RESERVED = 512L * 1024 * 1024;
static constexpr size_t REQUIRE_FREE_MEMORY_RESERVED_PER_SM = 40L * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_THRESHOLD = 10L * 1024 * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_OFFSET = 1L * 512 * 1024 * 1024;

using namespace o2::gpu;

__global__ void dummyInitKernel(void*) {}

#include "GPUReconstructionIncludesITS.h"

GPUReconstructionCUDA::GPUReconstructionCUDA(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionProcessing::KernelInterface<GPUReconstructionCUDA, GPUReconstructionDeviceBase>(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionCUDAInternals;
    *mParDevice = o2::gpu::internal::GPUDefParametersLoad();
  }
  mDeviceBackendSettings->deviceType = DeviceType::CUDA;
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
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

static_assert(sizeof(cudaError_t) <= sizeof(int64_t) && cudaSuccess == 0);
int32_t GPUReconstructionCUDA::GPUChkErrInternal(const int64_t error, const char* file, int32_t line) const
{
  return internal::GPUReconstructionCUDAChkErr(error, file, line);
}

GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCUDA(cfg); }

void GPUReconstructionCUDA::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits<7>>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits<7>>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame<7>>* timeFrame)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraitsGPU);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraits<7>); // TODO gpu-code to be implemented
  }
  if (timeFrame) {
    timeFrame->reset(new o2::its::gpu::TimeFrameGPU);
  }
}

int32_t GPUReconstructionCUDA::InitDevice_Runtime()
{
#ifndef __HIPCC__ // CUDA
  constexpr int32_t reqVerMaj = 2;
  constexpr int32_t reqVerMin = 0;
#endif
  if (GetProcessingSettings().rtc.enable && GetProcessingSettings().rtctech.runTest == 2) {
    mWarpSize = GPUCA_WARP_SIZE;
    genAndLoadRTC();
    exit(0);
  }

  if (mMaster == nullptr) {
    cudaDeviceProp deviceProp;
    int32_t count, bestDevice = -1;
    double bestDeviceSpeed = -1, deviceSpeed;
    if (GPUChkErrI(cudaGetDeviceCount(&count))) {
      GPUError("Error getting CUDA Device Count");
      return (1);
    }
    if (GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("Available CUDA devices:");
    }
    std::vector<bool> devicesOK(count, false);
    std::vector<size_t> devMemory(count, 0);
    std::vector<bool> contextCreated(count, false);
    for (int32_t i = 0; i < count; i++) {
      if (GetProcessingSettings().debugLevel >= 4) {
        GPUInfo("Examining device %d", i);
      }
      size_t free, total;
      if (GPUChkErrI(cudaSetDevice(i))) {
        if (GetProcessingSettings().debugLevel >= 4) {
          GPUWarning("Couldn't create context for device %d. Skipping it.", i);
        }
        continue;
      }
      contextCreated[i] = true;
      if (GPUChkErrI(cudaMemGetInfo(&free, &total))) {
        if (GetProcessingSettings().debugLevel >= 4) {
          GPUWarning("Error obtaining CUDA memory info about device %d! Skipping it.", i);
        }
        continue;
      }
      if (GetProcessingSettings().debugLevel >= 4) {
        GPUInfo("Obtained current memory usage for device %d", i);
      }
      if (GPUChkErrI(cudaGetDeviceProperties(&deviceProp, i))) {
        continue;
      }
      if (GetProcessingSettings().debugLevel >= 4) {
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
      if (GetProcessingSettings().debugLevel >= 2) {
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
        if (GetProcessingSettings().debugLevel >= 2 && GetProcessingSettings().deviceNum < 0) {
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
    } else if (GetProcessingSettings().deviceNum > -1) {
      if (GetProcessingSettings().deviceNum >= (signed)count) {
        GPUError("Requested device ID %d does not exist", GetProcessingSettings().deviceNum);
        noDevice = true;
      } else if (!devicesOK[GetProcessingSettings().deviceNum]) {
        GPUError("Unsupported device requested (%d)", GetProcessingSettings().deviceNum);
        noDevice = true;
      } else {
        bestDevice = GetProcessingSettings().deviceNum;
      }
    }
    for (int32_t i = 0; i < count; i++) {
      if (contextCreated[i] && (noDevice || i != bestDevice)) {
        GPUChkErrI(cudaSetDevice(i));
        GPUChkErrI(cudaDeviceReset());
      }
    }
    if (noDevice) {
      return (1);
    }
    mDeviceId = bestDevice;
    if (GPUChkErrI(cudaSetDevice(mDeviceId))) {
      GPUError("Could not set CUDA Device!");
      return (1);
    }

    GPUChkErrI(cudaGetDeviceProperties(&deviceProp, mDeviceId));

    if (GetProcessingSettings().debugLevel >= 2) {
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
    if (deviceProp.warpSize != GPUCA_WARP_SIZE && !GetProcessingSettings().rtc.enable) {
      throw std::runtime_error("Invalid warp size on GPU");
    }
    mWarpSize = deviceProp.warpSize;
    mBlockCount = deviceProp.multiProcessorCount;
    mMaxBackendThreads = std::max<int32_t>(mMaxBackendThreads, deviceProp.maxThreadsPerBlock * mBlockCount);
    mDeviceName = deviceProp.name;
    mDeviceName += " (CUDA GPU)";

    if (deviceProp.major < 3) {
      GPUError("Unsupported CUDA Device");
      return (1);
    }

#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (gGPUConstantMemBufferSize > deviceProp.totalConstMem) {
      GPUError("Insufficient constant memory available on GPU %d < %d!", (int32_t)deviceProp.totalConstMem, (int32_t)gGPUConstantMemBufferSize);
      return (1);
    }
#endif

#ifndef __HIPCC__ // CUDA
    if (GPUChkErrI(cudaDeviceSetLimit(cudaLimitStackSize, GPUCA_GPU_STACK_SIZE))) {
      GPUError("Error setting CUDA stack size");
      GPUChkErrI(cudaDeviceReset());
      return (1);
    }
    if (GPUChkErrI(cudaDeviceSetLimit(cudaLimitMallocHeapSize, GetProcessingSettings().deterministicGPUReconstruction ? std::max<size_t>(1024 * 1024 * 1024, GPUCA_GPU_HEAP_SIZE) : GPUCA_GPU_HEAP_SIZE))) {
      GPUError("Error setting CUDA stack size");
      GPUChkErrI(cudaDeviceReset());
      return (1);
    }
#else // HIP
    if (GPUChkErrI(hipSetDeviceFlags(hipDeviceScheduleBlockingSync))) {
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

    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Allocating memory on GPU");
    }
    if (mDeviceMemorySize > deviceProp.totalGlobalMem || GPUChkErrI(cudaMalloc(&mDeviceMemoryBase, mDeviceMemorySize))) {
      size_t free, total;
      GPUChkErr(cudaMemGetInfo(&free, &total));
      GPUError("CUDA Memory Allocation Error (trying %ld bytes, %ld available on GPU, %ld free)", (int64_t)mDeviceMemorySize, (int64_t)deviceProp.totalGlobalMem, (int64_t)free);
      GPUChkErrI(cudaDeviceReset());
      return (1);
    }
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Allocating memory on Host");
    }
    if (GPUChkErrI(cudaMallocHost(&mHostMemoryBase, mHostMemorySize))) {
      GPUError("Error allocating Page Locked Host Memory (trying %ld bytes)", (int64_t)mHostMemorySize);
      GPUChkErrI(cudaDeviceReset());
      return (1);
    }
    if (GetProcessingSettings().debugLevel >= 1) {
      GPUInfo("Memory ptrs: GPU (%ld bytes): %p - Host (%ld bytes): %p", (int64_t)mDeviceMemorySize, mDeviceMemoryBase, (int64_t)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
      if (GPUChkErrI(cudaMemset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize))) {
        GPUError("Error during CUDA memset");
        GPUChkErrI(cudaDeviceReset());
        return (1);
      }
    }

    for (int32_t i = 0; i < mNStreams; i++) {
      if (GPUChkErrI(cudaStreamCreateWithFlags(&mInternals->Streams[i], cudaStreamNonBlocking))) {
        GPUError("Error creating CUDA Stream");
        GPUChkErrI(cudaDeviceReset());
        return (1);
      }
    }

#ifndef __HIPCC__ // CUDA
    dummyInitKernel<<<mBlockCount, 256>>>(mDeviceMemoryBase);
#else // HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dummyInitKernel), dim3(mBlockCount), dim3(256), 0, 0, mDeviceMemoryBase);
#endif

    if (GetProcessingSettings().rtc.enable) {
      genAndLoadRTC();
    }
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
    else {
#ifdef GPUCA_RTC_NO_COMPILED_KERNELS
      GPUFatal("Compiled with GPUCA_RTC_NO_COMPILED_KERNELS, must run RTC mode!");
#endif
#define GPUCA_KRNL(x_class, ...)                                        \
  mInternals->kernelModules.emplace_back(std::make_unique<CUmodule>()); \
  GPUChkErr(cuModuleLoadData(mInternals->kernelModules.back().get(), GPUCA_M_CAT3(_binary_cuda_kernel_module_fatbin_krnl_, GPUCA_M_KRNL_NAME(x_class), GPUCA_M_CAT(PER_KERNEL_OBJECT_EXT, _start))));
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
      loadKernelModules(true);
    }
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
      GPUChkErr(cuModuleGetGlobal(&tmp, &tmpSize, *mInternals->kernelModules[i], "gGPUConstantMemBuffer"));
      mDeviceConstantMemList.emplace_back((void*)tmp);
    }
#else
    GPUChkErr(cudaMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize));
#endif
    mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;

    GPUInfo("CUDA Initialisation successfull (Device %d: %s (Frequency %d, Cores %d), %ld / %ld bytes host / global memory, Stack frame %d, Constant memory %ld)", mDeviceId, deviceProp.name, deviceProp.clockRate, deviceProp.multiProcessorCount, (int64_t)mHostMemorySize, (int64_t)mDeviceMemorySize, (int32_t)GPUCA_GPU_STACK_SIZE, (int64_t)gGPUConstantMemBufferSize);
  } else {
    GPUReconstructionCUDA* master = dynamic_cast<GPUReconstructionCUDA*>(mMaster);
    mDeviceId = master->mDeviceId;
    mBlockCount = master->mBlockCount;
    mWarpSize = master->mWarpSize;
    mMaxBackendThreads = master->mMaxBackendThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mDeviceConstantMemList.resize(master->mDeviceConstantMemList.size());
    std::copy(master->mDeviceConstantMemList.begin(), master->mDeviceConstantMemList.end(), mDeviceConstantMemList.begin());
    mInternals = master->mInternals;
    GPUChkErr(cudaSetDevice(mDeviceId));

    GPUInfo("CUDA Initialisation successfull (from master)");
  }

  for (uint32_t i = 0; i < mEvents.size(); i++) {
    cudaEvent_t* events = (cudaEvent_t*)mEvents[i].data();
    for (uint32_t j = 0; j < mEvents[i].size(); j++) {
#ifndef __HIPCC__ // CUDA
      if (GPUChkErrI(cudaEventCreate(&events[j]))) {
#else
      if (GPUChkErrI(hipEventCreateWithFlags(&events[j], hipEventBlockingSync))) {
#endif
        GPUError("Error creating event");
        GPUChkErrI(cudaDeviceReset());
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
    if (GetProcessingSettings().rtctech.runTest != 2) {
      mInternals->kernelModules.emplace_back(std::make_unique<CUmodule>());
      GPUChkErr(cuModuleLoad(mInternals->kernelModules.back().get(), (filename + "_" + std::to_string(i) + mRtcBinExtension).c_str()));
    }
    if (!GetProcessingSettings().rtctech.keepTempFiles) {
      remove((filename + "_" + std::to_string(i) + mRtcSrcExtension).c_str());
      remove((filename + "_" + std::to_string(i) + mRtcBinExtension).c_str());
    }
  }
  if (GetProcessingSettings().rtctech.runTest == 2) {
    return;
  }
  loadKernelModules(GetProcessingSettings().rtc.compilePerKernel);
}

int32_t GPUReconstructionCUDA::ExitDevice_Runtime()
{
  // Uninitialize CUDA
  GPUChkErr(cudaSetDevice(mDeviceId));
  SynchronizeGPU();
  unregisterRemainingRegisteredMemory();

  for (uint32_t i = 0; i < mEvents.size(); i++) {
    cudaEvent_t* events = (cudaEvent_t*)mEvents[i].data();
    for (uint32_t j = 0; j < mEvents[i].size(); j++) {
      GPUChkErrI(cudaEventDestroy(events[j]));
    }
  }

  if (mMaster == nullptr) {
    GPUChkErrI(cudaFree(mDeviceMemoryBase));
#ifdef GPUCA_NO_CONSTANT_MEMORY
    GPUChkErrI(cudaFree(mDeviceConstantMem));
#endif

    for (int32_t i = 0; i < mNStreams; i++) {
      GPUChkErrI(cudaStreamDestroy(mInternals->Streams[i]));
    }

    GPUChkErrI(cudaFreeHost(mHostMemoryBase));
    for (uint32_t i = 0; i < mInternals->kernelModules.size(); i++) {
      GPUChkErr(cuModuleUnload(*mInternals->kernelModules[i]));
    }

    GPUChkErrI(cudaDeviceReset());
    GPUInfo("CUDA Uninitialized");
  }
  mDeviceMemoryBase = nullptr;
  mHostMemoryBase = nullptr;

  return (0);
}

size_t GPUReconstructionCUDA::GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  if (GetProcessingSettings().debugLevel >= 3) {
    stream = -1;
  }
  if (stream == -1) {
    SynchronizeGPU();
    GPUChkErr(cudaMemcpy(dst, src, size, toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));
  } else {
    if (evList == nullptr) {
      nEvents = 0;
    }
    for (int32_t k = 0; k < nEvents; k++) {
      GPUChkErr(cudaStreamWaitEvent(mInternals->Streams[stream], evList[k].get<cudaEvent_t>(), 0));
    }
    GPUChkErr(cudaMemcpyAsync(dst, src, size, toGPU == -2 ? cudaMemcpyDeviceToDevice : (toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost), mInternals->Streams[stream]));
  }
  if (ev) {
    GPUChkErr(cudaEventRecord(ev->get<cudaEvent_t>(), mInternals->Streams[stream == -1 ? 0 : stream]));
  }
  if (GetProcessingSettings().serializeGPU & 2) {
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
      GPUChkErr(cudaMemcpy(((char*)basePtr) + offset, src, size, cudaMemcpyHostToDevice));
    } else {
      GPUChkErr(cudaMemcpyAsync(((char*)basePtr) + offset, src, size, cudaMemcpyHostToDevice, mInternals->Streams[stream]));
    }
  }
  if (ev && stream != -1) {
    GPUChkErr(cudaEventRecord(ev->get<cudaEvent_t>(), mInternals->Streams[stream]));
  }
  if (GetProcessingSettings().serializeGPU & 2) {
    GPUDebug("WriteToConstantMemory", stream, true);
  }
  return size;
}

void GPUReconstructionCUDA::ReleaseEvent(deviceEvent ev) {}
void GPUReconstructionCUDA::RecordMarker(deviceEvent* ev, int32_t stream) { GPUChkErr(cudaEventRecord(ev->get<cudaEvent_t>(), mInternals->Streams[stream])); }

std::unique_ptr<GPUReconstructionProcessing::threadContext> GPUReconstructionCUDA::GetThreadContext()
{
  GPUChkErr(cudaSetDevice(mDeviceId));
  return GPUReconstructionProcessing::GetThreadContext();
}

void GPUReconstructionCUDA::SynchronizeGPU() { GPUChkErr(cudaDeviceSynchronize()); }
void GPUReconstructionCUDA::SynchronizeStream(int32_t stream) { GPUChkErr(cudaStreamSynchronize(mInternals->Streams[stream])); }

void GPUReconstructionCUDA::SynchronizeEvents(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    GPUChkErr(cudaEventSynchronize(evList[i].get<cudaEvent_t>()));
  }
}

void GPUReconstructionCUDA::StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    GPUChkErr(cudaStreamWaitEvent(mInternals->Streams[stream], evList[i].get<cudaEvent_t>(), 0));
  }
}

bool GPUReconstructionCUDA::IsEventDone(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    cudaError_t retVal = cudaEventSynchronize(evList[i].get<cudaEvent_t>());
    if (retVal == cudaErrorNotReady) {
      return false;
    }
    GPUChkErr(retVal);
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
  if (!force && GetProcessingSettings().debugLevel <= 0) {
    return (0);
  }
  if (GPUChkErrI(stream == -1 ? cudaDeviceSynchronize() : cudaStreamSynchronize(mInternals->Streams[stream]))) {
    GPUError("CUDA Error while synchronizing (%s) (Stream %d)", state, stream);
    return (1);
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

int32_t GPUReconstructionCUDA::registerMemoryForGPU_internal(const void* ptr, size_t size)
{
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("Registering %zu bytes of memory for GPU", size);
  }
  return GPUChkErrI(cudaHostRegister((void*)ptr, size, cudaHostRegisterDefault));
}

int32_t GPUReconstructionCUDA::unregisterMemoryForGPU_internal(const void* ptr)
{
  return GPUChkErrI(cudaHostUnregister((void*)ptr));
}

void GPUReconstructionCUDA::PrintKernelOccupancies()
{
  int32_t maxBlocks = 0, threads = 0, suggestedBlocks = 0, nRegs = 0, sMem = 0;
  GPUChkErr(cudaSetDevice(mDeviceId));
  for (uint32_t i = 0; i < mInternals->kernelFunctions.size(); i++) {
    GPUChkErr(cuOccupancyMaxPotentialBlockSize(&suggestedBlocks, &threads, *mInternals->kernelFunctions[i], 0, 0, 0)); // NOLINT: failure in clang-tidy
    GPUChkErr(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, *mInternals->kernelFunctions[i], threads, 0));
    GPUChkErr(cuFuncGetAttribute(&nRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, *mInternals->kernelFunctions[i]));
    GPUChkErr(cuFuncGetAttribute(&sMem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *mInternals->kernelFunctions[i]));
    GPUInfo("Kernel: %50s Block size: %4d, Maximum active blocks: %3d, Suggested blocks: %3d, Regs: %3d, smem: %3d", GetKernelName(i).c_str(), threads, maxBlocks, suggestedBlocks, nRegs, sMem);
  }
}

void GPUReconstructionCUDA::loadKernelModules(bool perKernel)
{
  uint32_t j = 0;
#define GPUCA_KRNL(x_class, ...)                                                                                                                                                        \
  if (GetKernelNum<GPUCA_M_KRNL_TEMPLATE(x_class)>() != j) {                                                                                                                            \
    GPUFatal("kernel numbers out of sync");                                                                                                                                             \
  }                                                                                                                                                                                     \
  mInternals->kernelFunctions.emplace_back(new CUfunction);                                                                                                                             \
  if (GetProcessingSettings().debugLevel >= 3) {                                                                                                                                        \
    GPUInfo("Loading kernel %s (j = %u)", GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), j);                                                                              \
  }                                                                                                                                                                                     \
  GPUChkErr(cuModuleGetFunction(mInternals->kernelFunctions.back().get(), *mInternals->kernelModules[perKernel ? j : 0], GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))))); \
  j++;
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
  if (j != mInternals->kernelModules.size()) {
    GPUFatal("Did not load all kernels (%u < %u)", j, (uint32_t)mInternals->kernelModules.size());
  }
}

#define ORTCHK(command)                               \
  {                                                   \
    OrtStatus* status = command;                      \
    if (status != nullptr) {                          \
      const char* msg = api->GetErrorMessage(status); \
      GPUFatal("ONNXRuntime Error: %s", msg);         \
    }                                                 \
  }

void GPUReconstructionCUDA::SetONNXGPUStream(Ort::SessionOptions& session_options, int32_t stream, int32_t* deviceId)
{
  GPUChkErr(cudaGetDevice(deviceId));
#if !defined(__HIPCC__) && defined(ORT_CUDA_BUILD)
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  ORTCHK(api->CreateCUDAProviderOptions(&cuda_options));

  // std::vector<const char*> keys{"device_id", "gpu_mem_limit", "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream", "cudnn_conv_use_max_workspace", "cudnn_conv1d_pad_to_nc1d"};
  // std::vector<const char*> values{"0", "2147483648", "kSameAsRequested", "DEFAULT", "1", "1", "1"};
  // UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());

  // this implicitly sets "has_user_compute_stream"
  ORTCHK(api->UpdateCUDAProviderOptionsWithValue(cuda_options, "user_compute_stream", mInternals->Streams[stream]));
  ORTCHK(api->SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options));

  // Finally, don't forget to release the provider options
  api->ReleaseCUDAProviderOptions(cuda_options);
#elif defined(ORT_ROCM_BUILD)
  // const auto& api = Ort::GetApi();
  // api.GetCurrentGpuDeviceId(deviceId);
  OrtROCMProviderOptions rocm_options;
  rocm_options.has_user_compute_stream = 1; // Indicate that we are passing a user stream
  rocm_options.arena_extend_strategy = 0;   // kNextPowerOfTwo = 0, kSameAsRequested = 1 -> https://github.com/search?q=repo%3Amicrosoft%2Fonnxruntime%20kSameAsRequested&type=code
  // rocm_options.gpu_mem_limit = 1073741824; // 0 means no limit
  rocm_options.user_compute_stream = mInternals->Streams[stream];
  session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif // ORT_ROCM_BUILD
}

#ifndef __HIPCC__ // CUDA

void GPUReconstructionCUDA::startGPUProfiling()
{
  GPUChkErr(cudaProfilerStart());
}

void GPUReconstructionCUDA::endGPUProfiling()
{
  GPUChkErr(cudaProfilerStop());
}

#else // HIP
void* GPUReconstructionHIP::getGPUPointer(void* ptr)
{
  void* retVal = nullptr;
  GPUChkErr(hipHostGetDevicePointer(&retVal, ptr, 0));
  return retVal;
}

#endif // __HIPCC__
