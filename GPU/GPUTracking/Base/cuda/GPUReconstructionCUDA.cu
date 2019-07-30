// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDA.cu
/// \author David Rohr

#define GPUCA_GPUTYPE_PASCAL

#include <cuda.h>
#include <sm_20_atomic_functions.h>

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

constexpr size_t gGPUConstantMemBufferSize = (sizeof(GPUConstantMem) + sizeof(uint4) - 1);
#ifndef GPUCA_CUDA_NO_CONSTANT_MEMORY
__constant__ uint4 gGPUConstantMemBuffer[gGPUConstantMemBufferSize / sizeof(uint4)];
#define GPUCA_CONSMEM_PTR
#define GPUCA_CONSMEM_CALL
#define GPUCA_CONSMEM (GPUConstantMem&)gGPUConstantMemBuffer
#else
#define GPUCA_CONSMEM_PTR const uint4 *gGPUConstantMemBuffer,
#define GPUCA_CONSMEM_CALL (const uint4*)mDeviceConstantMem,
#define GPUCA_CONSMEM (GPUConstantMem&)(*gGPUConstantMemBuffer)
#endif

#ifdef GPUCA_USE_TEXTURES
texture<cahit2, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu2;
texture<calink, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu;
#endif

#ifdef HAVE_O2HEADERS
#include "ITStrackingCUDA/TrackerTraitsNV.h"
#include "ITStrackingCUDA/VertexerTraitsGPU.h"
#else
namespace o2
{
namespace its
{
class TrackerTraitsNV : public TrackerTraits
{
};
class VertexerTraitsGPU : public VertexerTraits
{
};
} // namespace its
} // namespace o2
#endif

#include "GPUReconstructionIncludesDevice.h"

template <class T, int I, typename... Args>
GPUg() void runKernelCUDA(GPUCA_CONSMEM_PTR int iSlice, Args... args)
{
  GPUshared() typename T::GPUTPCSharedMemory smem;
  T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Processor(GPUCA_CONSMEM)[iSlice], args...);
}

template <class T, int I, typename... Args>
GPUg() void runKernelCUDAMulti(GPUCA_CONSMEM_PTR int firstSlice, int nSliceCount, Args... args)
{
  const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
  const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
  const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
  const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
  GPUshared() typename T::GPUTPCSharedMemory smem;
  T::template Thread<I>(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, T::Processor(GPUCA_CONSMEM)[firstSlice + iSlice], args...);
}

template <class T, int I, typename... Args>
int GPUReconstructionCUDABackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
  if (z.evList) {
    for (int k = 0; k < z.nEvents; k++) {
      GPUFailedMsg(cudaStreamWaitEvent(mInternals->CudaStreams[x.stream], ((cudaEvent_t*)z.evList)[k], 0));
    }
  }
  if (y.num <= 1) {
    runKernelCUDA<T, I><<<x.nBlocks, x.nThreads, 0, mInternals->CudaStreams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, args...);
  } else {
    runKernelCUDAMulti<T, I><<<x.nBlocks, x.nThreads, 0, mInternals->CudaStreams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, y.num, args...);
  }
  GPUFailedMsg(cudaGetLastError());
  if (z.ev) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)z.ev, mInternals->CudaStreams[x.stream]));
  }
  return 0;
}

GPUReconstructionCUDABackend::GPUReconstructionCUDABackend(const GPUSettingsProcessing& cfg) : GPUReconstructionDeviceBase(cfg)
{
  mInternals = new GPUReconstructionCUDAInternals;
  mProcessingSettings.deviceType = DeviceType::CUDA;
}

GPUReconstructionCUDABackend::~GPUReconstructionCUDABackend()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit CUDA
  delete mInternals;
}

GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUSettingsProcessing& cfg) { return new GPUReconstructionCUDA(cfg); }

void GPUReconstructionCUDABackend::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraitsNV);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraitsGPU);
  }
}

int GPUReconstructionCUDABackend::InitDevice_Runtime()
{
  // Find best CUDA device, initialize and allocate memory
  cudaDeviceProp cudaDeviceProp;

  int count, bestDevice = -1;
  double bestDeviceSpeed = -1, deviceSpeed;
  if (GPUFailedMsgI(cudaGetDeviceCount(&count))) {
    GPUError("Error getting CUDA Device Count");
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Available CUDA devices:");
  }
  const int reqVerMaj = 2;
  const int reqVerMin = 0;
  std::vector<bool> devicesOK(count, false);
  for (int i = 0; i < count; i++) {
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      GPUInfo("Examining device %d", i);
    }
    size_t free, total;
    cuInit(0);
    CUdevice tmpDevice;
    cuDeviceGet(&tmpDevice, i);
    CUcontext tmpContext;
    cuCtxCreate(&tmpContext, 0, tmpDevice);
    if (cuMemGetInfo(&free, &total)) {
      std::cout << "Error\n";
    }
    cuCtxDestroy(tmpContext);
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      GPUInfo("Obtained current memory usage for device %d", i);
    }
    if (GPUFailedMsgI(cudaGetDeviceProperties(&cudaDeviceProp, i))) {
      continue;
    }
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      GPUInfo("Obtained device properties for device %d", i);
    }
    int deviceOK = true;
    const char* deviceFailure = "";
    if (cudaDeviceProp.major >= 9) {
      deviceOK = false;
      deviceFailure = "Invalid Revision";
    } else if (cudaDeviceProp.major < reqVerMaj || (cudaDeviceProp.major == reqVerMaj && cudaDeviceProp.minor < reqVerMin)) {
      deviceOK = false;
      deviceFailure = "Too low device revision";
    } else if (free < mDeviceMemorySize) {
      deviceOK = false;
      deviceFailure = "Insufficient GPU memory";
    }

    deviceSpeed = (double)cudaDeviceProp.multiProcessorCount * (double)cudaDeviceProp.clockRate * (double)cudaDeviceProp.warpSize * (double)free * (double)cudaDeviceProp.major * (double)cudaDeviceProp.major;
    if (mDeviceProcessingSettings.debugLevel >= 2) {
      GPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem Avail %lld / %lld)%s %s", deviceOK ? " " : "[", i, cudaDeviceProp.name, cudaDeviceProp.major, cudaDeviceProp.minor, (long long int)free, (long long int)cudaDeviceProp.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
    }
    if (!deviceOK) {
      continue;
    }
    devicesOK[i] = true;
    if (deviceSpeed > bestDeviceSpeed) {
      bestDevice = i;
      bestDeviceSpeed = deviceSpeed;
    } else {
      if (mDeviceProcessingSettings.debugLevel >= 0) {
        GPUInfo("Skipping: Speed %f < %f\n", deviceSpeed, bestDeviceSpeed);
      }
    }
  }
  if (bestDevice == -1) {
    GPUWarning("No %sCUDA Device available, aborting CUDA Initialisation", count ? "appropriate " : "");
    GPUImportant("Requiring Revision %d.%d, Mem: %lld", reqVerMaj, reqVerMin, (long long int)mDeviceMemorySize);
    return (1);
  }

  if (mDeviceProcessingSettings.deviceNum > -1) {
    if (mDeviceProcessingSettings.deviceNum >= (signed)count) {
      GPUWarning("Requested device ID %d does not exist", mDeviceProcessingSettings.deviceNum);
      return (1);
    } else if (!devicesOK[mDeviceProcessingSettings.deviceNum]) {
      GPUWarning("Unsupported device requested (%d)", mDeviceProcessingSettings.deviceNum);
      return (1);
    } else {
      bestDevice = mDeviceProcessingSettings.deviceNum;
    }
  }
  mDeviceId = bestDevice;

  GPUFailedMsgI(cudaGetDeviceProperties(&cudaDeviceProp, mDeviceId));

  if (mDeviceProcessingSettings.debugLevel >= 2) {
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
  mCoreCount = cudaDeviceProp.multiProcessorCount;
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
#ifndef GPUCA_CUDA_NO_CONSTANT_MEMORY
  if (gGPUConstantMemBufferSize > cudaDeviceProp.totalConstMem) {
    GPUError("Insufficient constant memory available on GPU %d < %d!", (int)cudaDeviceProp.totalConstMem, (int)gGPUConstantMemBufferSize);
    return (1);
  }
#endif

  mNStreams = std::max(mDeviceProcessingSettings.nStreams, 3);

  if (cuCtxCreate(&mInternals->CudaContext, CU_CTX_SCHED_AUTO, mDeviceId) != CUDA_SUCCESS) {
    GPUError("Could not set CUDA Device!");
    return (1);
  }

  if (GPUFailedMsgI(cudaDeviceSetLimit(cudaLimitStackSize, GPUCA_GPU_STACK_SIZE))) {
    GPUError("Error setting CUDA stack size");
    GPUFailedMsgI(cudaDeviceReset());
    return (1);
  }

  if (mDeviceMemorySize > cudaDeviceProp.totalGlobalMem || GPUFailedMsgI(cudaMalloc(&mDeviceMemoryBase, mDeviceMemorySize))) {
    GPUError("CUDA Memory Allocation Error");
    GPUFailedMsgI(cudaDeviceReset());
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 1) {
    GPUInfo("GPU Memory used: %lld (Ptr 0x%p)", (long long int)mDeviceMemorySize, mDeviceMemoryBase);
  }
  if (GPUFailedMsgI(cudaMallocHost(&mHostMemoryBase, mHostMemorySize))) {
    GPUError("Error allocating Page Locked Host Memory");
    GPUFailedMsgI(cudaDeviceReset());
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 1) {
    GPUInfo("Host Memory used: %lld (Ptr 0x%p)", (long long int)mHostMemorySize, mHostMemoryBase);
  }

  if (mDeviceProcessingSettings.debugLevel >= 1) {
    memset(mHostMemoryBase, 0xDD, mHostMemorySize);
    if (GPUFailedMsgI(cudaMemset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize))) {
      GPUError("Error during CUDA memset");
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
  }

  for (int i = 0; i < mNStreams; i++) {
    if (GPUFailedMsgI(cudaStreamCreate(&mInternals->CudaStreams[i]))) {
      GPUError("Error creating CUDA Stream");
      GPUFailedMsgI(cudaDeviceReset());
      return (1);
    }
  }

  void* devPtrConstantMem;
#ifndef GPUCA_CUDA_NO_CONSTANT_MEMORY
  if (GPUFailedMsgI(cudaGetSymbolAddress(&devPtrConstantMem, gGPUConstantMemBuffer))) {
    GPUError("Error getting ptr to constant memory");
    GPUFailedMsgI(cudaDeviceReset());
    return 1;
  }
#else
  if (GPUFailedMsgI(cudaMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize))) {
    GPUError("CUDA Memory Allocation Error");
    GPUFailedMsgI(cudaDeviceReset());
    return (1);
  }
#endif
  mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;

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

  cuCtxPopCurrent(&mInternals->CudaContext);
  GPUInfo("CUDA Initialisation successfull (Device %d: %s (Frequency %d, Cores %d), %lld / %lld bytes host / global memory, Stack frame %d, Constant memory %lld)", mDeviceId, cudaDeviceProp.name, cudaDeviceProp.clockRate, cudaDeviceProp.multiProcessorCount, (long long int)mHostMemorySize,
          (long long int)mDeviceMemorySize, (int)GPUCA_GPU_STACK_SIZE, (long long int)gGPUConstantMemBufferSize);

  return (0);
}

int GPUReconstructionCUDABackend::ExitDevice_Runtime()
{
  // Uninitialize CUDA
  cuCtxPushCurrent(mInternals->CudaContext);

  SynchronizeGPU();

  GPUFailedMsgI(cudaFree(mDeviceMemoryBase));
  mDeviceMemoryBase = nullptr;
#ifdef GPUCA_CUDA_NO_CONSTANT_MEMORY
  GPUFailedMsgI(cudaFree(mDeviceConstantMem));
#endif

  for (int i = 0; i < mNStreams; i++) {
    GPUFailedMsgI(cudaStreamDestroy(mInternals->CudaStreams[i]));
  }

  GPUFailedMsgI(cudaFreeHost(mHostMemoryBase));
  mHostMemoryBase = nullptr;

  for (unsigned int i = 0; i < mEvents.size(); i++) {
    cudaEvent_t* events = (cudaEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
      GPUFailedMsgI(cudaEventDestroy(events[j]));
    }
  }

  if (GPUFailedMsgI(cudaDeviceReset())) {
    GPUError("Could not uninitialize GPU");
    return (1);
  }

  cuCtxDestroy(mInternals->CudaContext);

  GPUInfo("CUDA Uninitialized");
  return (0);
}

void GPUReconstructionCUDABackend::GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
  if (mDeviceProcessingSettings.debugLevel >= 3) {
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
      GPUFailedMsg(cudaStreamWaitEvent(mInternals->CudaStreams[stream], ((cudaEvent_t*)evList)[k], 0));
    }
    GPUFailedMsg(cudaMemcpyAsync(dst, src, size, toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, mInternals->CudaStreams[stream]));
  }
  if (ev) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->CudaStreams[stream == -1 ? 0 : stream]));
  }
}

void GPUReconstructionCUDABackend::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst)
{
  if (!(res->Type() & GPUMemoryResource::MEMORY_GPU)) {
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      GPUInfo("Skipped transfer of non-GPU memory resource: %s", res->Name());
    }
    return;
  }
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    GPUInfo(toGPU ? "Copying to GPU: %s\n" : "Copying to Host: %s", res->Name());
  }
  GPUMemCpy(dst, src, res->Size(), stream, toGPU, ev, evList, nEvents);
}

void GPUReconstructionCUDABackend::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
#ifndef GPUCA_CUDA_NO_CONSTANT_MEMORY
  if (stream == -1) {
    GPUFailedMsg(cudaMemcpyToSymbol(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice));
  } else {
    GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice, mInternals->CudaStreams[stream]));
  }

#else
  if (stream == -1) {
    GPUFailedMsg(cudaMemcpy(((char*)mDeviceConstantMem) + offset, src, size, cudaMemcpyHostToDevice));
  } else {
    GPUFailedMsg(cudaMemcpyAsync(((char*)mDeviceConstantMem) + offset, src, size, cudaMemcpyHostToDevice, mInternals->CudaStreams[stream]));
  }

#endif
  if (ev && stream != -1) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->CudaStreams[stream]));
  }
}

void GPUReconstructionCUDABackend::ReleaseEvent(deviceEvent* ev) {}
void GPUReconstructionCUDABackend::RecordMarker(deviceEvent* ev, int stream) { GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->CudaStreams[stream])); }

GPUReconstructionCUDABackend::GPUThreadContextCUDA::GPUThreadContextCUDA(GPUReconstructionCUDAInternals* context) : GPUThreadContext(), mContext(context) { cuCtxPushCurrent(mContext->CudaContext); }
GPUReconstructionCUDABackend::GPUThreadContextCUDA::~GPUThreadContextCUDA() { cuCtxPopCurrent(&mContext->CudaContext); }
std::unique_ptr<GPUReconstruction::GPUThreadContext> GPUReconstructionCUDABackend::GetThreadContext() { return std::unique_ptr<GPUThreadContext>(new GPUThreadContextCUDA(mInternals)); }

void GPUReconstructionCUDABackend::SynchronizeGPU() { GPUFailedMsg(cudaDeviceSynchronize()); }
void GPUReconstructionCUDABackend::SynchronizeStream(int stream) { GPUFailedMsg(cudaStreamSynchronize(mInternals->CudaStreams[stream])); }

void GPUReconstructionCUDABackend::SynchronizeEvents(deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(cudaEventSynchronize(((cudaEvent_t*)evList)[i]));
  }
}

bool GPUReconstructionCUDABackend::IsEventDone(deviceEvent* evList, int nEvents)
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

int GPUReconstructionCUDABackend::GPUDebug(const char* state, int stream)
{
  // Wait for CUDA-Kernel to finish and check for CUDA errors afterwards, in case of debugmode
  if (mDeviceProcessingSettings.debugLevel == 0) {
    return (0);
  }
  cudaError cuErr;
  cuErr = cudaGetLastError();
  if (cuErr != cudaSuccess) {
    GPUError("Cuda Error %s while running kernel (%s) (Stream %d)", cudaGetErrorString(cuErr), state, stream);
    return (1);
  }
  if (GPUFailedMsgI(cudaDeviceSynchronize())) {
    GPUError("CUDA Error while synchronizing (%s) (Stream %d)", state, stream);
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

int GPUReconstructionCUDABackend::PrepareTextures()
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

void GPUReconstructionCUDABackend::SetThreadCounts()
{
  mThreadCount = GPUCA_THREAD_COUNT;
  mBlockCount = mCoreCount;
  mConstructorBlockCount = mBlockCount * (mDeviceProcessingSettings.trackletConstructorInPipeline ? 1 : GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER);
  mSelectorBlockCount = mBlockCount * GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER;
  mConstructorThreadCount = GPUCA_THREAD_COUNT_CONSTRUCTOR;
  mSelectorThreadCount = GPUCA_THREAD_COUNT_SELECTOR;
  mFinderThreadCount = GPUCA_THREAD_COUNT_FINDER;
  mTRDThreadCount = GPUCA_THREAD_COUNT_TRD;
}
