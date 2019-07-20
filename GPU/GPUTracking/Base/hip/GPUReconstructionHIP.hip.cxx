// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionHIP.hip.cxx
/// \author David Rohr

#include "hip/hip_runtime.h"
#define GPUCA_GPUTYPE_HIP

#include "GPUReconstructionHIP.h"
#include "GPUReconstructionHIPInternals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

constexpr size_t gGPUConstantMemBufferSize = (sizeof(GPUConstantMem) + sizeof(uint4) - 1);
#ifndef GPUCA_HIP_NO_CONSTANT_MEMORY
__constant__ uint4 gGPUConstantMemBuffer[gGPUConstantMemBufferSize / sizeof(uint4)];
__global__ void gGPUConstantMemBuffer_dummy(uint4* p) { p[0] = gGPUConstantMemBuffer[0]; }
#define GPUCA_CONSMEM_PTR
#define GPUCA_CONSMEM_CALL
#define GPUCA_CONSMEM (GPUConstantMem&)gGPUConstantMemBuffer
#else
#define GPUCA_CONSMEM_PTR const uint4 *gGPUConstantMemBuffer,
#define GPUCA_CONSMEM_CALL (const uint4*)mDeviceConstantMem,
#define GPUCA_CONSMEM (GPUConstantMem&)(*gGPUConstantMemBuffer)
#endif

namespace o2
{
namespace its
{
class TrackerTraitsHIP : public TrackerTraits
{
};
} // namespace its
} // namespace o2

#include "GPUReconstructionIncludesDevice.h"

template <class T, int I, typename... Args>
GPUg() void runKernelHIP(GPUCA_CONSMEM_PTR int iSlice, Args... args)
{
  GPUshared() typename T::GPUTPCSharedMemory smem;
  T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Processor(GPUCA_CONSMEM)[iSlice], args...);
}

template <class T, int I, typename... Args>
GPUg() void runKernelHIPMulti(GPUCA_CONSMEM_PTR int firstSlice, int nSliceCount, Args... args)
{
  const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
  const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
  const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
  const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
  GPUshared() typename T::GPUTPCSharedMemory smem;
  T::template Thread<I>(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, T::Processor(GPUCA_CONSMEM)[firstSlice + iSlice], args...);
}

template <class T, int I, typename... Args>
int GPUReconstructionHIPBackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, Args... args)
{
  if (z.evList) {
    for (int k = 0; k < z.nEvents; k++) {
      GPUFailedMsg(hipStreamWaitEvent(mInternals->HIPStreams[x.stream], ((hipEvent_t*)z.evList)[k], 0));
    }
  }
  if (y.num <= 1) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIP<T, I, Args...>), dim3(x.nBlocks), dim3(x.nThreads), 0, mInternals->HIPStreams[x.stream], GPUCA_CONSMEM_CALL y.start, args...);
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIPMulti<T, I, Args...>), dim3(x.nBlocks), dim3(x.nThreads), 0, mInternals->HIPStreams[x.stream], GPUCA_CONSMEM_CALL y.start, y.num, args...);
  }
  if (z.ev) {
    GPUFailedMsg(hipEventRecord(*(hipEvent_t*)z.ev, mInternals->HIPStreams[x.stream]));
  }
  return 0;
}

GPUReconstructionHIPBackend::GPUReconstructionHIPBackend(const GPUSettingsProcessing& cfg) : GPUReconstructionDeviceBase(cfg)
{
  mInternals = new GPUReconstructionHIPInternals;
  mProcessingSettings.deviceType = DeviceType::HIP;
}

GPUReconstructionHIPBackend::~GPUReconstructionHIPBackend()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit CUDA
  delete mInternals;
}

GPUReconstruction* GPUReconstruction_Create_HIP(const GPUSettingsProcessing& cfg) { return new GPUReconstructionHIP(cfg); }

int GPUReconstructionHIPBackend::InitDevice_Runtime()
{
  // Find best HIP device, initialize and allocate memory

  hipDeviceProp_t hipDeviceProp_t;

  int count, bestDevice = -1;
  double bestDeviceSpeed = -1, deviceSpeed;
  if (GPUFailedMsgI(hipGetDeviceCount(&count))) {
    GPUError("Error getting HIP Device Count");
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Available HIP devices:");
  }
  const int reqVerMaj = 2;
  const int reqVerMin = 0;
  for (int i = 0; i < count; i++) {
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      printf("Examining device %d\n", i);
    }
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      printf("Obtained current memory usage for device %d\n", i);
    }
    if (GPUFailedMsgI(hipGetDeviceProperties(&hipDeviceProp_t, i))) {
      continue;
    }
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      printf("Obtained device properties for device %d\n", i);
    }
    int deviceOK = true;
    const char* deviceFailure = "";
    if (hipDeviceProp_t.major >= 9) {
      deviceOK = false;
      deviceFailure = "Invalid Revision";
    } else if (hipDeviceProp_t.major < reqVerMaj || (hipDeviceProp_t.major == reqVerMaj && hipDeviceProp_t.minor < reqVerMin)) {
      deviceOK = false;
      deviceFailure = "Too low device revision";
    }

    deviceSpeed = (double)hipDeviceProp_t.multiProcessorCount * (double)hipDeviceProp_t.clockRate * (double)hipDeviceProp_t.warpSize * (double)hipDeviceProp_t.major * (double)hipDeviceProp_t.major;
    if (mDeviceProcessingSettings.debugLevel >= 2) {
      GPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem %lld)%s %s", deviceOK ? " " : "[", i, hipDeviceProp_t.name, hipDeviceProp_t.major, hipDeviceProp_t.minor, (long long int)hipDeviceProp_t.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
    }
    if (!deviceOK) {
      continue;
    }
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
    GPUWarning("No %sHIP Device available, aborting HIP Initialisation", count ? "appropriate " : "");
    GPUImportant("Requiring Revision %d.%d, Mem: %lld", reqVerMaj, reqVerMin, (long long int)mDeviceMemorySize);
    return (1);
  }

  if (mDeviceProcessingSettings.deviceNum > -1) {
    if (mDeviceProcessingSettings.deviceNum < (signed)count) {
      bestDevice = mDeviceProcessingSettings.deviceNum;
    } else {
      GPUWarning("Requested device ID %d non existend, falling back to default device id %d", mDeviceProcessingSettings.deviceNum, bestDevice);
    }
  }
  mDeviceId = bestDevice;

  GPUFailedMsgI(hipGetDeviceProperties(&hipDeviceProp_t, mDeviceId));

  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Using HIP Device %s with Properties:", hipDeviceProp_t.name);
    GPUInfo("\ttotalGlobalMem = %lld", (unsigned long long int)hipDeviceProp_t.totalGlobalMem);
    GPUInfo("\tsharedMemPerBlock = %lld", (unsigned long long int)hipDeviceProp_t.sharedMemPerBlock);
    GPUInfo("\tregsPerBlock = %d", hipDeviceProp_t.regsPerBlock);
    GPUInfo("\twarpSize = %d", hipDeviceProp_t.warpSize);
    GPUInfo("\tmaxThreadsPerBlock = %d", hipDeviceProp_t.maxThreadsPerBlock);
    GPUInfo("\tmaxThreadsDim = %d %d %d", hipDeviceProp_t.maxThreadsDim[0], hipDeviceProp_t.maxThreadsDim[1], hipDeviceProp_t.maxThreadsDim[2]);
    GPUInfo("\tmaxGridSize = %d %d %d", hipDeviceProp_t.maxGridSize[0], hipDeviceProp_t.maxGridSize[1], hipDeviceProp_t.maxGridSize[2]);
    GPUInfo("\ttotalConstMem = %lld", (unsigned long long int)hipDeviceProp_t.totalConstMem);
    GPUInfo("\tmajor = %d", hipDeviceProp_t.major);
    GPUInfo("\tminor = %d", hipDeviceProp_t.minor);
    GPUInfo("\tclockRate = %d", hipDeviceProp_t.clockRate);
    GPUInfo("\tmemoryClockRate = %d", hipDeviceProp_t.memoryClockRate);
    GPUInfo("\tmultiProcessorCount = %d", hipDeviceProp_t.multiProcessorCount);
    GPUInfo(" ");
  }
  mCoreCount = hipDeviceProp_t.multiProcessorCount;
  mDeviceName = hipDeviceProp_t.name;
  mDeviceName += " (HIP GPU)";

  if (hipDeviceProp_t.major < 3) {
    GPUError("Unsupported HIP Device");
    return (1);
  }
#ifndef GPUCA_HIP_NO_CONSTANT_MEMORY
  if (gGPUConstantMemBufferSize > hipDeviceProp_t.totalConstMem) {
    GPUError("Insufficient constant memory available on GPU %d < %d!", (int)hipDeviceProp_t.totalConstMem, (int)gGPUConstantMemBufferSize);
    return (1);
  }
#endif

  mNStreams = std::max(mDeviceProcessingSettings.nStreams, 3);

  /*if (GPUFailedMsgI(hipDeviceSetLimit(hipLimitStackSize, GPUCA_GPU_STACK_SIZE)))
  {
    GPUError("Error setting HIP stack size");
    GPUFailedMsgI(hipDeviceReset());
    return(1);
  }*/

  if (mDeviceMemorySize > hipDeviceProp_t.totalGlobalMem) {
    GPUError("Insufficient GPU memory (%lld < %lld)", (long long int)hipDeviceProp_t.totalGlobalMem, (long long int)mDeviceMemorySize);
    GPUFailedMsgI(hipDeviceReset());
    return (1);
  }
  if (GPUFailedMsgI(hipMalloc(&mDeviceMemoryBase, mDeviceMemorySize))) {
    GPUError("HIP Memory Allocation Error");
    GPUFailedMsgI(hipDeviceReset());
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 1) {
    GPUInfo("GPU Memory used: %lld (Ptr 0x%p)", (long long int)mDeviceMemorySize, mDeviceMemoryBase);
  }
  if (GPUFailedMsgI(hipHostMalloc(&mHostMemoryBase, mHostMemorySize))) {
    GPUError("Error allocating Page Locked Host Memory");
    GPUFailedMsgI(hipDeviceReset());
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 1) {
    GPUInfo("Host Memory used: %lld (Ptr 0x%p)", (long long int)mHostMemorySize, mHostMemoryBase);
  }

  if (mDeviceProcessingSettings.debugLevel >= 1) {
    memset(mHostMemoryBase, 0, mHostMemorySize);
    if (GPUFailedMsgI(hipMemset(mDeviceMemoryBase, 143, mDeviceMemorySize))) {
      GPUError("Error during HIP memset");
      GPUFailedMsgI(hipDeviceReset());
      return (1);
    }
  }

  for (int i = 0; i < mNStreams; i++) {
    if (GPUFailedMsgI(hipStreamCreate(&mInternals->HIPStreams[i]))) {
      GPUError("Error creating HIP Stream");
      GPUFailedMsgI(hipDeviceReset());
      return (1);
    }
  }

  void* devPtrConstantMem;
#ifndef GPUCA_HIP_NO_CONSTANT_MEMORY
  if (GPUFailedMsgI(hipGetSymbolAddress(&devPtrConstantMem, HIP_SYMBOL(gGPUConstantMemBuffer)))) {
    GPUError("Error getting ptr to constant memory");
    GPUFailedMsgI(hipDeviceReset());
    return 1;
  }
#else
  if (GPUFailedMsgI(hipMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize))) {
    GPUError("HIP Memory Allocation Error");
    GPUFailedMsgI(hipDeviceReset());
    return (1);
  }
#endif
  mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;

  for (unsigned int i = 0; i < mEvents.size(); i++) {
    hipEvent_t* events = (hipEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
      if (GPUFailedMsgI(hipEventCreate(&events[j]))) {
        GPUError("Error creating event");
        GPUFailedMsgI(hipDeviceReset());
        return 1;
      }
    }
  }

  GPUInfo("HIP Initialisation successfull (Device %d: %s (Frequency %d, Cores %d), %'lld / %'lld bytes host / global memory, Stack frame %'d, Constant memory %'lld)", mDeviceId, hipDeviceProp_t.name, hipDeviceProp_t.clockRate, hipDeviceProp_t.multiProcessorCount, (long long int)mHostMemorySize,
          (long long int)mDeviceMemorySize, (int)GPUCA_GPU_STACK_SIZE, (long long int)gGPUConstantMemBufferSize);

  return (0);
}

int GPUReconstructionHIPBackend::ExitDevice_Runtime()
{
  // Uninitialize HIP
  SynchronizeGPU();

  GPUFailedMsgI(hipFree(mDeviceMemoryBase));
  mDeviceMemoryBase = nullptr;
#ifdef GPUCA_HIP_NO_CONSTANT_MEMORY
  GPUFailedMsgI(hipFree(mDeviceConstantMem));
#endif

  for (int i = 0; i < mNStreams; i++) {
    GPUFailedMsgI(hipStreamDestroy(mInternals->HIPStreams[i]));
  }

  GPUFailedMsgI(hipHostFree(mHostMemoryBase));
  mHostMemoryBase = nullptr;

  for (unsigned int i = 0; i < mEvents.size(); i++) {
    hipEvent_t* events = (hipEvent_t*)mEvents[i].data();
    for (unsigned int j = 0; j < mEvents[i].size(); j++) {
      GPUFailedMsgI(hipEventDestroy(events[j]));
    }
  }

  if (GPUFailedMsgI(hipDeviceReset())) {
    GPUError("Could not uninitialize GPU");
    return (1);
  }

  GPUInfo("HIP Uninitialized");
  return (0);
}

void GPUReconstructionHIPBackend::GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
  if (mDeviceProcessingSettings.debugLevel >= 3) {
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
      GPUFailedMsg(hipStreamWaitEvent(mInternals->HIPStreams[stream], ((hipEvent_t*)evList)[k], 0));
    }
    GPUFailedMsg(hipMemcpyAsync(dst, src, size, toGPU ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost, mInternals->HIPStreams[stream]));
  }
  if (ev) {
    GPUFailedMsg(hipEventRecord(*(hipEvent_t*)ev, mInternals->HIPStreams[stream == -1 ? 0 : stream]));
  }
}

void GPUReconstructionHIPBackend::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst)
{
  if (!(res->Type() & GPUMemoryResource::MEMORY_GPU)) {
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      printf("Skipped transfer of non-GPU memory resource: %s\n", res->Name());
    }
    return;
  }
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    printf(toGPU ? "Copying to GPU: %s\n" : "Copying to Host: %s\n", res->Name());
  }
  GPUMemCpy(dst, src, res->Size(), stream, toGPU, ev, evList, nEvents);
}

void GPUReconstructionHIPBackend::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
#ifndef GPUCA_HIP_NO_CONSTANT_MEMORY
  if (stream == -1) {
    GPUFailedMsg(hipMemcpyToSymbol(gGPUConstantMemBuffer, src, size, offset, hipMemcpyHostToDevice));
  } else {
    GPUFailedMsg(hipMemcpyToSymbolAsync(gGPUConstantMemBuffer, src, size, offset, hipMemcpyHostToDevice, mInternals->HIPStreams[stream]));
  }
  if (ev && stream != -1) {
    GPUFailedMsg(hipEventRecord(*(hipEvent_t*)ev, mInternals->HIPStreams[stream]));
  }

#else
  if (stream == -1) {
    GPUFailedMsg(hipMemcpy(((char*)mDeviceConstantMem) + offset, src, size, hipMemcpyHostToDevice));
  } else {
    GPUFailedMsg(hipMemcpyAsync(((char*)mDeviceConstantMem) + offset, src, size, hipMemcpyHostToDevice, mInternals->HIPStreams[stream]));
  }

#endif
}

void GPUReconstructionHIPBackend::ReleaseEvent(deviceEvent* ev) {}

void GPUReconstructionHIPBackend::RecordMarker(deviceEvent* ev, int stream) { GPUFailedMsg(hipEventRecord(*(hipEvent_t*)ev, mInternals->HIPStreams[stream])); }

void GPUReconstructionHIPBackend::SynchronizeGPU() { GPUFailedMsg(hipDeviceSynchronize()); }

void GPUReconstructionHIPBackend::SynchronizeStream(int stream) { GPUFailedMsg(hipStreamSynchronize(mInternals->HIPStreams[stream])); }

void GPUReconstructionHIPBackend::SynchronizeEvents(deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(hipEventSynchronize(((hipEvent_t*)evList)[i]));
  }
}

bool GPUReconstructionHIPBackend::IsEventDone(deviceEvent* evList, int nEvents)
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

int GPUReconstructionHIPBackend::GPUDebug(const char* state, int stream)
{
  // Wait for HIP-Kernel to finish and check for HIP errors afterwards, in case of debugmode
  if (mDeviceProcessingSettings.debugLevel == 0) {
    return (0);
  }
  hipError_t cuErr;
  cuErr = hipGetLastError();
  if (cuErr != hipSuccess) {
    GPUError("HIP Error %s while running kernel (%s) (Stream %d)", hipGetErrorString(cuErr), state, stream);
    return (1);
  }
  if (GPUFailedMsgI(hipDeviceSynchronize())) {
    GPUError("HIP Error while synchronizing (%s) (Stream %d)", state, stream);
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

void GPUReconstructionHIPBackend::SetThreadCounts()
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
