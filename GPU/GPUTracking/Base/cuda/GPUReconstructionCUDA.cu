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

static constexpr size_t REQUIRE_MIN_MEMORY = 1024L * 1024 * 1024;
static constexpr size_t REQUIRE_MEMORY_RESERVED = 512L * 1024 * 1024;
static constexpr size_t REQUIRE_FREE_MEMORY_RESERVED_PER_SM = 40L * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_THRESHOLD = 10L * 1024 * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_OFFSET = 1L * 512 * 1024 * 1024;

using namespace GPUCA_NAMESPACE::gpu;

__global__ void dummyInitKernel(void*)
{
}

#if defined(GPUCA_HAVE_O2HEADERS) && !defined(GPUCA_NO_ITS_TRAITS)
#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/VertexerTraitsGPU.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#else
namespace o2::its
{
class VertexerTraitsGPU : public VertexerTraits
{
};
template <int NLayers = 7>
class TrackerTraitsGPU : public TrackerTraits
{
};
namespace gpu
{
template <int NLayers = 7>
class TimeFrameGPU : public TimeFrame
{
};
} // namespace gpu
} // namespace o2::its
#endif

GPUReconstructionCUDABackend::GPUReconstructionCUDABackend(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionDeviceBase(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionCUDAInternals;
  }
}

GPUReconstructionCUDABackend::~GPUReconstructionCUDABackend()
{
  if (mMaster == nullptr) {
    for (unsigned int i = 0; i < mInternals->rtcModules.size(); i++) {
      cuModuleUnload(*mInternals->rtcModules[i]);
    }
    delete mInternals;
  }
}

GPUReconstructionCUDA::GPUReconstructionCUDA(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionKernels(cfg)
{
  mDeviceBackendSettings.deviceType = DeviceType::CUDA;
}

GPUReconstructionCUDA::~GPUReconstructionCUDA()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit CUDA
}

GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCUDA(cfg); }

void GPUReconstructionCUDA::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraitsGPU);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraitsGPU);
  }
}

void GPUReconstructionCUDA::GetITSTimeframe(std::unique_ptr<o2::its::TimeFrame>* timeFrame)
{
  timeFrame->reset(new o2::its::gpu::TimeFrameGPU);
}

void GPUReconstructionCUDA::UpdateSettings()
{
  GPUCA_GPUReconstructionUpdateDefailts();
}

int GPUReconstructionCUDA::InitDevice_Runtime()
{
  if (mMaster == nullptr) {
    cudaDeviceProp cudaDeviceProp;
    int count, bestDevice = -1;
    double bestDeviceSpeed = -1, deviceSpeed;
    if (GPUFailedMsgI(cuInit(0))) {
      GPUError("Error initializing CUDA!");
      return (1);
    }
    if (GPUFailedMsgI(cudaGetDeviceCount(&count))) {
      GPUError("Error getting CUDA Device Count");
      return (1);
    }
    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Available CUDA devices:");
    }
    const int reqVerMaj = 2;
    const int reqVerMin = 0;
    std::vector<bool> devicesOK(count, false);
    std::vector<size_t> devMemory(count, 0);
    bool contextCreated = false;
    for (int i = 0; i < count; i++) {
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Examining device %d", i);
      }
      size_t free, total;
      CUdevice tmpDevice;
      if (GPUFailedMsgI(cuDeviceGet(&tmpDevice, i))) {
        GPUError("Could not set CUDA device!");
        return (1);
      }
      if (GPUFailedMsgI(cuCtxCreate(&mInternals->CudaContext, 0, tmpDevice))) {
        if (mProcessingSettings.debugLevel >= 4) {
          GPUWarning("Couldn't create context for device %d. Skipping it.", i);
        }
        continue;
      }
      contextCreated = true;
      if (GPUFailedMsgI(cuMemGetInfo(&free, &total))) {
        if (mProcessingSettings.debugLevel >= 4) {
          GPUWarning("Error obtaining CUDA memory info about device %d! Skipping it.", i);
        }
        GPUFailedMsg(cuCtxDestroy(mInternals->CudaContext));
        continue;
      }
      if (count > 1) {
        GPUFailedMsg(cuCtxDestroy(mInternals->CudaContext));
        contextCreated = false;
      }
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Obtained current memory usage for device %d", i);
      }
      if (GPUFailedMsgI(cudaGetDeviceProperties(&cudaDeviceProp, i))) {
        continue;
      }
      if (mProcessingSettings.debugLevel >= 4) {
        GPUInfo("Obtained device properties for device %d", i);
      }
      int deviceOK = true;
      const char* deviceFailure = "";
      if (cudaDeviceProp.major < reqVerMaj || (cudaDeviceProp.major == reqVerMaj && cudaDeviceProp.minor < reqVerMin)) {
        deviceOK = false;
        deviceFailure = "Too low device revision";
      } else if (free < std::max<size_t>(mDeviceMemorySize, REQUIRE_MIN_MEMORY)) {
        deviceOK = false;
        deviceFailure = "Insufficient GPU memory";
      }

      deviceSpeed = (double)cudaDeviceProp.multiProcessorCount * (double)cudaDeviceProp.clockRate * (double)cudaDeviceProp.warpSize * (double)free * (double)cudaDeviceProp.major * (double)cudaDeviceProp.major;
      if (mProcessingSettings.debugLevel >= 2) {
        GPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem Avail %lu / %lu)%s %s", deviceOK ? " " : "[", i, cudaDeviceProp.name, cudaDeviceProp.major, cudaDeviceProp.minor, free, (size_t)cudaDeviceProp.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
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
      GPUWarning("No %sCUDA Device available, aborting CUDA Initialisation", count ? "appropriate " : "");
      GPUImportant("Requiring Revision %d.%d, Mem: %lu", reqVerMaj, reqVerMin, std::max<size_t>(mDeviceMemorySize, REQUIRE_MIN_MEMORY));
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
        GPUFailedMsgI(cuCtxDestroy(mInternals->CudaContext));
      }
      return (1);
    }
    mDeviceId = bestDevice;

    GPUFailedMsgI(cudaGetDeviceProperties(&cudaDeviceProp, mDeviceId));

    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Using CUDA Device %s with Properties:", cudaDeviceProp.name);
      GPUInfo("\ttotalGlobalMem = %lld", (unsigned long long int)cudaDeviceProp.totalGlobalMem);
      GPUInfo("\tsharedMemPerBlock = %lld", (unsigned long long int)cudaDeviceProp.sharedMemPerBlock);
      GPUInfo("\tregsPerBlock = %d", cudaDeviceProp.regsPerBlock);
      GPUInfo("\twarpSize = %d", cudaDeviceProp.warpSize);
      GPUInfo("\tmemPitch = %lld", (unsigned long long int)cudaDeviceProp.memPitch);
      GPUInfo("\tmaxThreadsPerBlock = %d", cudaDeviceProp.maxThreadsPerBlock);
      GPUInfo("\tmaxThreadsDim = %d %d %d", cudaDeviceProp.maxThreadsDim[0], cudaDeviceProp.maxThreadsDim[1], cudaDeviceProp.maxThreadsDim[2]);
      GPUInfo("\tmaxGridSize = %d %d %d", cudaDeviceProp.maxGridSize[0], cudaDeviceProp.maxGridSize[1], cudaDeviceProp.maxGridSize[2]);
      GPUInfo("\ttotalConstMem = %lld", (unsigned long long int)cudaDeviceProp.totalConstMem);
      GPUInfo("\tmajor = %d", cudaDeviceProp.major);
      GPUInfo("\tminor = %d", cudaDeviceProp.minor);
      GPUInfo("\tclockRate = %d", cudaDeviceProp.clockRate);
      GPUInfo("\tmemoryClockRate = %d", cudaDeviceProp.memoryClockRate);
      GPUInfo("\tmultiProcessorCount = %d", cudaDeviceProp.multiProcessorCount);
      GPUInfo("\ttextureAlignment = %lld", (unsigned long long int)cudaDeviceProp.textureAlignment);
      GPUInfo(" ");
    }
    if (cudaDeviceProp.warpSize != GPUCA_WARP_SIZE) {
      throw std::runtime_error("Invalid warp size on GPU");
    }
    mBlockCount = cudaDeviceProp.multiProcessorCount;
    mWarpSize = 32;
    mMaxThreads = std::max<int>(mMaxThreads, cudaDeviceProp.maxThreadsPerBlock * mBlockCount);
    mDeviceName = cudaDeviceProp.name;
    mDeviceName += " (CUDA GPU)";

    if (cudaDeviceProp.major < 3) {
      GPUError("Unsupported CUDA Device");
      return (1);
    }

#ifdef GPUCA_USE_TEXTURES
    if (GPUCA_SLICE_DATA_MEMORY * NSLICES > (size_t)cudaDeviceProp.maxTexture1DLinear) {
      GPUError("Invalid maximum texture size of device: %lld < %lld\n", (long long int)cudaDeviceProp.maxTexture1DLinear, (long long int)(GPUCA_SLICE_DATA_MEMORY * NSLICES));
      return (1);
    }
#endif
#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (gGPUConstantMemBufferSize > cudaDeviceProp.totalConstMem) {
      GPUError("Insufficient constant memory available on GPU %d < %d!", (int)cudaDeviceProp.totalConstMem, (int)gGPUConstantMemBufferSize);
      return (1);
    }
#endif

    if (contextCreated == 0 && GPUFailedMsgI(cuCtxCreate(&mInternals->CudaContext, CU_CTX_SCHED_AUTO, mDeviceId))) {
      GPUError("Could not set CUDA Device!");
      return (1);
    }

    if (GPUFailedMsgI(cudaDeviceSetLimit(cudaLimitStackSize, GPUCA_GPU_STACK_SIZE))) {
      GPUError("Error setting CUDA stack size");
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
    if (GPUFailedMsgI(cudaDeviceSetLimit(cudaLimitMallocHeapSize, GPUCA_GPU_HEAP_SIZE))) {
      GPUError("Error setting CUDA stack size");
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }

    if (mDeviceMemorySize == 1 || mDeviceMemorySize == 2) {
      mDeviceMemorySize = std::max<long int>(0, devMemory[mDeviceId] - REQUIRE_FREE_MEMORY_RESERVED_PER_SM * cudaDeviceProp.multiProcessorCount); // Take all GPU memory but some reserve
      if (mDeviceMemorySize >= RESERVE_EXTRA_MEM_THRESHOLD) {
        mDeviceMemorySize -= RESERVE_EXTRA_MEM_OFFSET;
      }
    }
    if (mDeviceMemorySize == 2) {
      mDeviceMemorySize = mDeviceMemorySize * 2 / 3; // Leave 1/3 of GPU memory for event display
    }

    if (mDeviceMemorySize > cudaDeviceProp.totalGlobalMem || GPUFailedMsgI(cudaMalloc(&mDeviceMemoryBase, mDeviceMemorySize))) {
      GPUError("CUDA Memory Allocation Error (trying %lld bytes, %lld available)", (long long int)mDeviceMemorySize, (long long int)cudaDeviceProp.totalGlobalMem);
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
    if (GPUFailedMsgI(cudaMallocHost(&mHostMemoryBase, mHostMemorySize))) {
      GPUError("Error allocating Page Locked Host Memory (trying %lld bytes)", (long long int)mHostMemorySize);
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
    if (mProcessingSettings.debugLevel >= 1) {
      GPUInfo("Memory ptrs: GPU (%lld bytes): %p - Host (%lld bytes): %p", (long long int)mDeviceMemorySize, mDeviceMemoryBase, (long long int)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
      if (GPUFailedMsgI(cudaMemset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize))) {
        GPUError("Error during CUDA memset");
        GPUFailedMsgI(cudaDeviceReset());
        return (1);
      }
    }

    for (int i = 0; i < mNStreams; i++) {
      if (GPUFailedMsgI(cudaStreamCreateWithFlags(&mInternals->Streams[i], cudaStreamNonBlocking))) {
        GPUError("Error creating CUDA Stream");
        GPUFailedMsgI(cudaDeviceReset());
        return (1);
      }
    }

    dummyInitKernel<<<mBlockCount, 256>>>(mDeviceMemoryBase);
    GPUInfo("CUDA Initialisation successfull (Device %d: %s (Frequency %d, Cores %d), %lld / %lld bytes host / global memory, Stack frame %d, Constant memory %lld)", mDeviceId, cudaDeviceProp.name, cudaDeviceProp.clockRate, cudaDeviceProp.multiProcessorCount, (long long int)mHostMemorySize,
            (long long int)mDeviceMemorySize, (int)GPUCA_GPU_STACK_SIZE, (long long int)gGPUConstantMemBufferSize);

#ifndef GPUCA_ALIROOT_LIB
    if (mProcessingSettings.rtc.enable) {
      if (genRTC()) {
        throw std::runtime_error("Runtime compilation failed");
      }
    }
#endif
    void* devPtrConstantMem;
    if (mProcessingSettings.rtc.enable) {
      mDeviceConstantMemRTC.resize(mInternals->rtcModules.size());
    }
#ifndef GPUCA_NO_CONSTANT_MEMORY
    devPtrConstantMem = GetBackendConstSymbolAddress();
    if (mProcessingSettings.rtc.enable) {
      for (unsigned int i = 0; i < mDeviceConstantMemRTC.size(); i++) {
        GPUFailedMsg(cuModuleGetGlobal((CUdeviceptr*)&mDeviceConstantMemRTC[i], nullptr, *mInternals->rtcModules[i], "gGPUConstantMemBuffer"));
      }
    }
#else
    GPUFailedMsg(cudaMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize));
    for (unsigned int i = 0; i < mDeviceConstantMemRTC.size(); i++) {
      mDeviceConstantMemRTC[i] = devPtrConstantMem;
    }
#endif
    mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;
  } else {
    GPUReconstructionCUDA* master = dynamic_cast<GPUReconstructionCUDA*>(mMaster);
    mDeviceId = master->mDeviceId;
    mBlockCount = master->mBlockCount;
    mWarpSize = master->mWarpSize;
    mMaxThreads = master->mMaxThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mDeviceConstantMemRTC.resize(master->mDeviceConstantMemRTC.size());
    std::copy(master->mDeviceConstantMemRTC.begin(), master->mDeviceConstantMemRTC.end(), mDeviceConstantMemRTC.begin());
    mInternals = master->mInternals;
    GPUFailedMsgI(cuCtxPushCurrent(mInternals->CudaContext));
  }

  if (mProcessingSettings.debugLevel >= 1) {
  }
  for (unsigned int i = 0; i < mEvents.size(); i++) {
    cudaEvent_t* events = (cudaEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
      if (GPUFailedMsgI(cudaEventCreate(&events[j]))) {
        GPUError("Error creating event");
        GPUFailedMsgI(cudaDeviceReset());
        return 1;
      }
    }
  }

  if (GPUFailedMsgI(cuCtxPopCurrent(&mInternals->CudaContext))) {
    GPUError("Error popping CUDA context!");
    return (1);
  }

  return (0);
}

int GPUReconstructionCUDA::ExitDevice_Runtime()
{
  // Uninitialize CUDA
  GPUFailedMsgI(cuCtxPushCurrent(mInternals->CudaContext));

  SynchronizeGPU();
  for (unsigned int i = 0; i < mEvents.size(); i++) {
    cudaEvent_t* events = (cudaEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
      GPUFailedMsgI(cudaEventDestroy(events[j]));
    }
  }

  if (mMaster == nullptr) {
    GPUFailedMsgI(cudaFree(mDeviceMemoryBase));
#ifdef GPUCA_NO_CONSTANT_MEMORY
    GPUFailedMsgI(cudaFree(mDeviceConstantMem));
#endif

    for (int i = 0; i < mNStreams; i++) {
      GPUFailedMsgI(cudaStreamDestroy(mInternals->Streams[i]));
    }

    GPUFailedMsgI(cudaFreeHost(mHostMemoryBase));
    GPUFailedMsgI(cuCtxDestroy(mInternals->CudaContext));
    GPUInfo("CUDA Uninitialized");
  } else {
    GPUFailedMsgI(cuCtxPopCurrent(&mInternals->CudaContext));
  }
  mDeviceMemoryBase = nullptr;
  mHostMemoryBase = nullptr;

  /*if (GPUFailedMsgI(cudaDeviceReset())) { // No longer doing this, another thread might have used the GPU
    GPUError("Could not uninitialize GPU");
    return (1);
  }*/

  return (0);
}

size_t GPUReconstructionCUDA::GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
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
    for (int k = 0; k < nEvents; k++) {
      GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[stream], ((cudaEvent_t*)evList)[k], 0));
    }
    GPUFailedMsg(cudaMemcpyAsync(dst, src, size, toGPU == -2 ? cudaMemcpyDeviceToDevice : toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, mInternals->Streams[stream]));
  }
  if (ev) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->Streams[stream == -1 ? 0 : stream]));
  }
  return size;
}

size_t GPUReconstructionCUDA::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst)
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

size_t GPUReconstructionCUDA::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
  std::unique_ptr<GPUParamRTC> tmpParam;
  for (unsigned int i = 0; i < 1 + mDeviceConstantMemRTC.size(); i++) {
    void* basePtr = i ? mDeviceConstantMemRTC[i - 1] : mDeviceConstantMem;
    if (i && basePtr == (void*)mDeviceConstantMem) {
      continue;
    }
    if (stream == -1) {
      GPUFailedMsg(cudaMemcpy(((char*)basePtr) + offset, src, size, cudaMemcpyHostToDevice));
    } else {
      GPUFailedMsg(cudaMemcpyAsync(((char*)basePtr) + offset, src, size, cudaMemcpyHostToDevice, mInternals->Streams[stream]));
    }
  }
  if (ev && stream != -1) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->Streams[stream]));
  }
  return size;
}

void GPUReconstructionCUDA::ReleaseEvent(deviceEvent* ev) {}
void GPUReconstructionCUDA::RecordMarker(deviceEvent* ev, int stream) { GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->Streams[stream])); }

GPUReconstructionCUDA::GPUThreadContextCUDA::GPUThreadContextCUDA(GPUReconstructionCUDAInternals* context) : GPUThreadContext(), mContext(context)
{
  if (mContext->cudaContextObtained++ == 0) {
    cuCtxPushCurrent(mContext->CudaContext);
  }
}
GPUReconstructionCUDA::GPUThreadContextCUDA::~GPUThreadContextCUDA()
{
  if (--mContext->cudaContextObtained == 0) {
    cuCtxPopCurrent(&mContext->CudaContext);
  }
}
std::unique_ptr<GPUReconstruction::GPUThreadContext> GPUReconstructionCUDA::GetThreadContext() { return std::unique_ptr<GPUThreadContext>(new GPUThreadContextCUDA(mInternals)); }

void GPUReconstructionCUDA::SynchronizeGPU() { GPUFailedMsg(cudaDeviceSynchronize()); }
void GPUReconstructionCUDA::SynchronizeStream(int stream) { GPUFailedMsg(cudaStreamSynchronize(mInternals->Streams[stream])); }

void GPUReconstructionCUDA::SynchronizeEvents(deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(cudaEventSynchronize(((cudaEvent_t*)evList)[i]));
  }
}

void GPUReconstructionCUDA::StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[stream], ((cudaEvent_t*)evList)[i], 0));
  }
}

bool GPUReconstructionCUDA::IsEventDone(deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    cudaError_t retVal = cudaEventSynchronize(((cudaEvent_t*)evList)[i]);
    if (retVal == cudaErrorNotReady) {
      return false;
    }
    GPUFailedMsg(retVal);
  }
  return (true);
}

int GPUReconstructionCUDA::GPUDebug(const char* state, int stream, bool force)
{
  // Wait for CUDA-Kernel to finish and check for CUDA errors afterwards, in case of debugmode
  cudaError cuErr;
  cuErr = cudaGetLastError();
  if (cuErr != cudaSuccess) {
    GPUError("Cuda Error %s while running kernel (%s) (Stream %d)", cudaGetErrorString(cuErr), state, stream);
    return (1);
  }
  if (force == false && mProcessingSettings.debugLevel <= 0) {
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

int GPUReconstructionCUDA::PrepareTextures()
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

int GPUReconstructionCUDA::registerMemoryForGPU(const void* ptr, size_t size)
{
  return mProcessingSettings.noGPUMemoryRegistration ? 0 : GPUFailedMsgI(cudaHostRegister((void*)ptr, size, cudaHostRegisterDefault));
}

int GPUReconstructionCUDA::unregisterMemoryForGPU(const void* ptr)
{
  return mProcessingSettings.noGPUMemoryRegistration ? 0 : GPUFailedMsgI(cudaHostUnregister((void*)ptr));
}

void GPUReconstructionCUDA::startGPUProfiling()
{
  GPUFailedMsg(cudaProfilerStart());
}

void GPUReconstructionCUDA::endGPUProfiling()
{
  GPUFailedMsg(cudaProfilerStop());
}
