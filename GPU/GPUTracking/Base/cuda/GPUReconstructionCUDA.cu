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

#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionCUDAIncludes.h"

#include <cuda_profiler_api.h>
#include <unistd.h>

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "GPUReconstructionIncludes.h"
#include "GPUParamRTC.h"

static constexpr size_t REQUIRE_MIN_MEMORY = 1024L * 1024 * 1024;
static constexpr size_t REQUIRE_MEMORY_RESERVED = 512L * 1024 * 1024;
static constexpr size_t REQUIRE_FREE_MEMORY_RESERVED_PER_SM = 40L * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_THRESHOLD = 10L * 1024 * 1024 * 1024;
static constexpr size_t RESERVE_EXTRA_MEM_OFFSET = 1L * 512 * 1024 * 1024;

using namespace GPUCA_NAMESPACE::gpu;

#ifdef GPUCA_USE_TEXTURES
texture<cahit2, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu2;
texture<calink, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu;
#endif

__global__ void dummyInitKernel(void*)
{
}

#if defined(HAVE_O2HEADERS) && !defined(GPUCA_NO_ITS_TRAITS)
#include "ITStrackingCUDA/TrackerTraitsNV.h"
#include "ITStrackingCUDA/VertexerTraitsGPU.h"
#else
namespace o2::its
{
class TrackerTraitsNV : public TrackerTraits
{
};
class VertexerTraitsGPU : public VertexerTraits
{
};
} // namespace o2::its
#endif

class GPUDebugTiming
{
 public:
  GPUDebugTiming(bool d, void** t, cudaStream_t* s, GPUReconstruction::krnlSetup& x, GPUReconstructionCUDABackend* r = nullptr) : mDeviceTimers(t), mStreams(s), mXYZ(x), mRec(r), mDo(d)
  {
    if (mDo) {
      if (mDeviceTimers) {
        GPUFailedMsg(cudaEventRecord((cudaEvent_t)mDeviceTimers[0], mStreams[mXYZ.x.stream]));
      } else {
        mTimer.ResetStart();
      }
    }
  }
  ~GPUDebugTiming()
  {
    if (mDo) {
      if (mDeviceTimers) {
        GPUFailedMsg(cudaEventRecord((cudaEvent_t)mDeviceTimers[1], mStreams[mXYZ.x.stream]));
        GPUFailedMsg(cudaEventSynchronize((cudaEvent_t)mDeviceTimers[1]));
        float v;
        GPUFailedMsg(cudaEventElapsedTime(&v, (cudaEvent_t)mDeviceTimers[0], (cudaEvent_t)mDeviceTimers[1]));
        mXYZ.t = v * 1.e-3;
      } else {
        GPUFailedMsg(cudaStreamSynchronize(mStreams[mXYZ.x.stream]));
        mXYZ.t = mTimer.GetCurrentElapsedTime();
      }
    }
  }

 private:
  void** mDeviceTimers;
  cudaStream_t* mStreams;
  GPUReconstruction::krnlSetup& mXYZ;
  GPUReconstructionCUDABackend* mRec;
  HighResTimer mTimer;
  bool mDo;
};

#include "GPUReconstructionIncludesDevice.h"

#ifndef GPUCA_ALIROOT_LIB
extern "C" char _curtc_GPUReconstructionCUDArtc_cu_src[];
extern "C" unsigned int _curtc_GPUReconstructionCUDArtc_cu_src_size;
extern "C" char _curtc_GPUReconstructionCUDArtc_cu_command[];
#endif

/*
// Not using templated kernel any more, since nvidia profiler does not resolve template names
template <class T, int I, typename... Args>
GPUg() void runKernelCUDA(GPUCA_CONSMEM_PTR int iSlice_internal, Args... args)
{
  GPUshared() typename T::GPUSharedMemory smem;
  T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Processor(GPUCA_CONSMEM)[iSlice_internal], args...);
}
*/

#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_PROP(x_class, x_attributes)                          \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_CALL_single(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, args...);
#define GPUCA_KRNL_CALL_multi(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, y.num, args...);

#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL

template <bool multi, class T, int I>
int GPUReconstructionCUDAInternals::getRTCkernelNum(int k)
{
  static int num = k;
  if (num < 0) {
    throw std::runtime_error("Invalid kernel");
  }
  return num;
}

template <>
void GPUReconstructionCUDABackend::runKernelBackendInternal<GPUMemClean16, 0>(krnlSetup& _xyz, void* const& ptr, unsigned long const& size)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  GPUFailedMsg(cudaMemsetAsync(ptr, 0, size, mInternals->Streams[_xyz.x.stream]));
}

static void getArgPtrs(const void** pArgs) {}
template <typename T, typename... Args>
static void getArgPtrs(const void** pArgs, const T& arg, const Args&... args)
{
  *pArgs = &arg;
  getArgPtrs(pArgs + 1, args...);
}

template <class T, int I, typename... Args>
void GPUReconstructionCUDABackend::runKernelBackendInternal(krnlSetup& _xyz, const Args&... args)
{
  GPUDebugTiming timer(mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0, (void**)mDebugEvents, mInternals->Streams, _xyz);
  if (mProcessingSettings.enableRTC) {
    auto& x = _xyz.x;
    auto& y = _xyz.y;
    if (y.num <= 1) {
      const void* pArgs[sizeof...(Args) + 1];
      pArgs[0] = &y.start;
      getArgPtrs(&pArgs[1], args...);
      GPUFailedMsg(cuLaunchKernel(*mInternals->rtcFunctions[mInternals->getRTCkernelNum<false, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    } else {
      const void* pArgs[sizeof...(Args) + 2];
      pArgs[0] = &y.start;
      pArgs[1] = &y.num;
      getArgPtrs(&pArgs[2], args...);
      GPUFailedMsg(cuLaunchKernel(*mInternals->rtcFunctions[mInternals->getRTCkernelNum<true, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    }
  } else {
    backendInternal<T, I>::runKernelBackendMacro(_xyz, this, args...);
  }
}

template <class T, int I, typename... Args>
int GPUReconstructionCUDABackend::runKernelBackend(krnlSetup& _xyz, const Args&... args)
{
  auto& x = _xyz.x;
  auto& z = _xyz.z;
  if (z.evList) {
    for (int k = 0; k < z.nEvents; k++) {
      GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[x.stream], ((cudaEvent_t*)z.evList)[k], 0));
    }
  }
  runKernelBackendInternal<T, I>(_xyz, args...);
  GPUFailedMsg(cudaGetLastError());
  if (z.ev) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)z.ev, mInternals->Streams[x.stream]));
  }
  return 0;
}

GPUReconstructionCUDABackend::GPUReconstructionCUDABackend(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionDeviceBase(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionCUDAInternals;
  }
  mDeviceBackendSettings.deviceType = DeviceType::CUDA;
}

GPUReconstructionCUDABackend::~GPUReconstructionCUDABackend()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit CUDA
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCUDA(cfg); }

void GPUReconstructionCUDABackend::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraitsNV);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraitsGPU);
  }
}

void GPUReconstructionCUDABackend::UpdateSettings()
{
  GPUCA_GPUReconstructionUpdateDefailts();
}

int GPUReconstructionCUDABackend::InitDevice_Runtime()
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
    if (mProcessingSettings.enableRTC) {
      if (mProcessingSettings.debugLevel >= 0) {
        GPUInfo("Starting CUDA RTC Compilation");
      }
      HighResTimer rtcTimer;
      rtcTimer.ResetStart();
      std::string filename = "/tmp/o2cagpu_rtc_";
      filename += std::to_string(getpid());
      filename += "_";
      filename += std::to_string(rand());
      if (mProcessingSettings.debugLevel >= 3) {
        printf("Writing to %s\n", filename.c_str());
      }
      FILE* fp = fopen((filename + ".cu").c_str(), "w+b");
      if (fp == nullptr) {
        throw std::runtime_error("Error opening file");
      }
      std::string rtcparam = GPUParamRTC::generateRTCCode(param(), mProcessingSettings.rtcConstexpr);
      if (fwrite(rtcparam.c_str(), 1, rtcparam.size(), fp) != rtcparam.size()) {
        throw std::runtime_error("Error writing file");
      }
      if (fwrite(_curtc_GPUReconstructionCUDArtc_cu_src, 1, _curtc_GPUReconstructionCUDArtc_cu_src_size, fp) != _curtc_GPUReconstructionCUDArtc_cu_src_size) {
        throw std::runtime_error("Error writing file");
      }
      fclose(fp);
      std::string command = _curtc_GPUReconstructionCUDArtc_cu_command;
      command += " -cubin -c " + filename + ".cu -o " + filename + ".o";
      if (mProcessingSettings.debugLevel >= 3) {
        printf("Running command %s\n", command.c_str());
      }
      if (system(command.c_str())) {
        throw std::runtime_error("Runtime compilation failed");
      }
      GPUFailedMsg(cuModuleLoad(&mInternals->rtcModule, (filename + ".o").c_str()));

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward)                          \
  mInternals->getRTCkernelNum<false, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->rtcFunctions.size()); \
  mInternals->rtcFunctions.emplace_back(new CUfunction);                                               \
  GPUFailedMsg(cuModuleGetFunction(mInternals->rtcFunctions.back().get(), mInternals->rtcModule, GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class)))));
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward)                          \
  mInternals->getRTCkernelNum<true, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->rtcFunctions.size()); \
  mInternals->rtcFunctions.emplace_back(new CUfunction);                                              \
  GPUFailedMsg(cuModuleGetFunction(mInternals->rtcFunctions.back().get(), mInternals->rtcModule, GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi))));
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

      remove((filename + ".cu").c_str());
      remove((filename + ".o").c_str());
      if (mProcessingSettings.debugLevel >= 0) {
        GPUInfo("RTC Compilation finished (%f seconds)", rtcTimer.GetCurrentElapsedTime());
      }
    }
#endif
    void* devPtrConstantMem;
#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (mProcessingSettings.enableRTC) {
      GPUFailedMsg(cuModuleGetGlobal((CUdeviceptr*)&devPtrConstantMem, nullptr, mInternals->rtcModule, "gGPUConstantMemBuffer"));
    } else {
      GPUFailedMsg(cudaGetSymbolAddress(&devPtrConstantMem, gGPUConstantMemBuffer));
    }
#else
    GPUFailedMsg(cudaMalloc(&devPtrConstantMem, gGPUConstantMemBufferSize));
#endif
    mDeviceConstantMem = (GPUConstantMem*)devPtrConstantMem;
  } else {
    GPUReconstructionCUDABackend* master = dynamic_cast<GPUReconstructionCUDABackend*>(mMaster);
    mDeviceId = master->mDeviceId;
    mBlockCount = master->mBlockCount;
    mWarpSize = master->mWarpSize;
    mMaxThreads = master->mMaxThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
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

int GPUReconstructionCUDABackend::ExitDevice_Runtime()
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

size_t GPUReconstructionCUDABackend::GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
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

size_t GPUReconstructionCUDABackend::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst)
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

size_t GPUReconstructionCUDABackend::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
#ifndef GPUCA_NO_CONSTANT_MEMORY
  if (stream == -1) {
    GPUFailedMsg(cudaMemcpyToSymbol(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice));
  } else {
    GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice, mInternals->Streams[stream]));
  }
  if (mProcessingSettings.enableRTC)
#endif
  {
    std::unique_ptr<GPUParamRTC> tmpParam;
    if (mProcessingSettings.rtcConstexpr) {
      if (offset < sizeof(GPUParam) && (offset != 0 || size > sizeof(GPUParam))) {
        throw std::runtime_error("Invalid write to constant memory, crossing GPUParam border");
      }
      if (offset == 0) {
        tmpParam.reset(new GPUParamRTC);
        tmpParam->setFrom(*(GPUParam*)src);
        src = tmpParam.get();
        size = sizeof(*tmpParam);
      } else {
        offset = offset - sizeof(GPUParam) + sizeof(GPUParamRTC);
      }
    }
    if (stream == -1) {
      GPUFailedMsg(cudaMemcpy(((char*)mDeviceConstantMem) + offset, src, size, cudaMemcpyHostToDevice));
    } else {
      GPUFailedMsg(cudaMemcpyAsync(((char*)mDeviceConstantMem) + offset, src, size, cudaMemcpyHostToDevice, mInternals->Streams[stream]));
    }
  }
  if (ev && stream != -1) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->Streams[stream]));
  }
  return size;
}

void GPUReconstructionCUDABackend::ReleaseEvent(deviceEvent* ev) {}
void GPUReconstructionCUDABackend::RecordMarker(deviceEvent* ev, int stream) { GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)ev, mInternals->Streams[stream])); }

GPUReconstructionCUDABackend::GPUThreadContextCUDA::GPUThreadContextCUDA(GPUReconstructionCUDAInternals* context) : GPUThreadContext(), mContext(context)
{
  if (mContext->cudaContextObtained++ == 0) {
    cuCtxPushCurrent(mContext->CudaContext);
  }
}
GPUReconstructionCUDABackend::GPUThreadContextCUDA::~GPUThreadContextCUDA()
{
  if (--mContext->cudaContextObtained == 0) {
    cuCtxPopCurrent(&mContext->CudaContext);
  }
}
std::unique_ptr<GPUReconstruction::GPUThreadContext> GPUReconstructionCUDABackend::GetThreadContext() { return std::unique_ptr<GPUThreadContext>(new GPUThreadContextCUDA(mInternals)); }

void GPUReconstructionCUDABackend::SynchronizeGPU() { GPUFailedMsg(cudaDeviceSynchronize()); }
void GPUReconstructionCUDABackend::SynchronizeStream(int stream) { GPUFailedMsg(cudaStreamSynchronize(mInternals->Streams[stream])); }

void GPUReconstructionCUDABackend::SynchronizeEvents(deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(cudaEventSynchronize(((cudaEvent_t*)evList)[i]));
  }
}

void GPUReconstructionCUDABackend::StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents)
{
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[stream], ((cudaEvent_t*)evList)[i], 0));
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
  cudaError cuErr;
  cuErr = cudaGetLastError();
  if (cuErr != cudaSuccess) {
    GPUError("Cuda Error %s while running kernel (%s) (Stream %d)", cudaGetErrorString(cuErr), state, stream);
    return (1);
  }
  if (mProcessingSettings.debugLevel <= 0) {
    return (0);
  }
  if (GPUFailedMsgI(cudaDeviceSynchronize())) {
    GPUError("CUDA Error while synchronizing (%s) (Stream %d)", state, stream);
    return (1);
  }
  if (mProcessingSettings.debugLevel >= 3) {
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

int GPUReconstructionCUDABackend::registerMemoryForGPU(const void* ptr, size_t size)
{
  return GPUFailedMsgI(cudaHostRegister((void*)ptr, size, cudaHostRegisterDefault));
}

int GPUReconstructionCUDABackend::unregisterMemoryForGPU(const void* ptr)
{
  return GPUFailedMsgI(cudaHostUnregister((void*)ptr));
}

void GPUReconstructionCUDABackend::PrintKernelOccupancies()
{
  int maxBlocks, threads, suggestedBlocks;
  cudaFuncAttributes attr;
  GPUFailedMsg(cuCtxPushCurrent(mInternals->CudaContext));
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward)                                                          \
  GPUFailedMsg(cudaOccupancyMaxPotentialBlockSize(&suggestedBlocks, &threads, GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))));        \
  GPUFailedMsg(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class)), threads, 0)); \
  GPUFailedMsg(cudaFuncGetAttributes(&attr, GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))));                                          \
  GPUInfo("Kernel: %50s Block size: %4d, Maximum active blocks: %3d, Suggested blocks: %3d, Regs: %3d, smem: %3d", GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), threads, maxBlocks, suggestedBlocks, attr.numRegs, (int)attr.sharedSizeBytes);
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward)                                                                    \
  GPUFailedMsg(cudaOccupancyMaxPotentialBlockSize(&suggestedBlocks, &threads, GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)));        \
  GPUFailedMsg(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi), threads, 0)); \
  GPUFailedMsg(cudaFuncGetAttributes(&attr, GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)));                                          \
  GPUInfo("Kernel: %50s Block size: %4d, Maximum active blocks: %3d, Suggested blocks: %3d, Regs: %3d, smem: %3d", GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)), threads, maxBlocks, suggestedBlocks, attr.numRegs, (int)attr.sharedSizeBytes);
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi
  GPUFailedMsg(cuCtxPopCurrent(&mInternals->CudaContext));
}

void GPUReconstructionCUDABackend::startGPUProfiling()
{
  GPUFailedMsg(cudaProfilerStart());
}

void GPUReconstructionCUDABackend::endGPUProfiling()
{
  GPUFailedMsg(cudaProfilerStop());
}
