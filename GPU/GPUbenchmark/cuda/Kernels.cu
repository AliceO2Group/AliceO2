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
///
/// \file Kernels.{cu, hip.cxx}
/// \author: mconcas@cern.ch

#include "../Shared/Kernels.h"
#include <cstdio>

// Memory partitioning legend
//
// |----------------------region 0-----------------|----------------------region 1-----------------| regions -> deafult: 2, to test lower and upper RAM
// |--chunk 0--|--chunk 1--|--chunk 2--|                  ***                          |--chunk n--| chunks  -> default size: 1GB (sing block pins)
// |__________________________________________scratch______________________________________________| scratch -> default size: 95% free GPU RAM

#define GPUCHECK(error)                                                                        \
  if (error != cudaSuccess) {                                                                  \
    printf("%serror: '%s'(%d) at %s:%d%s\n", KRED, cudaGetErrorString(error), error, __FILE__, \
           __LINE__, KNRM);                                                                    \
    failed("API returned error code.");                                                        \
  }

double bytesToconfig(size_t s) { return (double)s / (1024.0); }
double bytesToGB(size_t s) { return (double)s / GB; }

// CUDA does not support <type4> operations:
// https://forums.developer.nvidia.com/t/swizzling-float4-arithmetic-support/217
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(int4& a, int4 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
#endif

namespace o2
{
namespace benchmark
{

namespace gpu
{
////////////
// Kernels

// Read
template <class chunk_t>
__global__ void read_k(
  chunk_t* chunkPtr,
  size_t chunkSize)
{
  chunk_t sink{0};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    sink += chunkPtr[i];
  }
  chunkPtr[threadIdx.x] = sink;
}

// Write
template <class chunk_t>
__global__ void write_k(
  chunk_t* chunkPtr,
  size_t chunkSize)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    chunkPtr[i] = 0;
  }
}

template <>
__global__ void write_k(
  int4* chunkPtr,
  size_t chunkSize)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    chunkPtr[i] = {0, 1, 0, 0};
  };
}

// Copy
template <class chunk_t>
__global__ void copy_k(
  chunk_t* chunkPtr,
  size_t chunkSize)
{
  size_t offset = chunkSize / 2;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < offset; i += blockDim.x * gridDim.x) {
    chunkPtr[i] = chunkPtr[offset + i];
  }
}

// Random read
template <class chunk_t>
__global__ void rand_read_k(
  chunk_t* chunkPtr,
  size_t chunkSize)
{
  chunk_t sink{0};
  BSDRnd r{};
  for (size_t i = threadIdx.x; i < chunkSize; i += blockDim.x) {
    sink = chunkPtr[i];
  }
  chunkPtr[threadIdx.x] = sink; // writing done once
}
} // namespace gpu

void printDeviceProp(int deviceId)
{
  const int w1 = 34;
  std::cout << std::left;
  std::cout << std::setw(w1)
            << "--------------------------------------------------------------------------------"
            << std::endl;
  std::cout << std::setw(w1) << "device#" << deviceId << std::endl;

  cudaDeviceProp props;
  GPUCHECK(cudaGetDeviceProperties(&props, deviceId));

  std::cout << std::setw(w1) << "Name: " << props.name << std::endl;
  std::cout << std::setw(w1) << "pciBusID: " << props.pciBusID << std::endl;
  std::cout << std::setw(w1) << "pciDeviceID: " << props.pciDeviceID << std::endl;
  std::cout << std::setw(w1) << "pciDomainID: " << props.pciDomainID << std::endl;
  std::cout << std::setw(w1) << "multiProcessorCount: " << props.multiProcessorCount << std::endl;
  std::cout << std::setw(w1) << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor
            << std::endl;
  std::cout << std::setw(w1) << "isMultiGpuBoard: " << props.isMultiGpuBoard << std::endl;
  std::cout << std::setw(w1) << "clockRate: " << (float)props.clockRate / 1000.0 << " Mhz" << std::endl;
  std::cout << std::setw(w1) << "memoryClockRate: " << (float)props.memoryClockRate / 1000.0 << " Mhz"
            << std::endl;
  std::cout << std::setw(w1) << "memoryBusWidth: " << props.memoryBusWidth << std::endl;
  std::cout << std::setw(w1) << "clockInstructionRate: " << (float)props.clockRate / 1000.0
            << " Mhz" << std::endl;
  std::cout << std::setw(w1) << "totalGlobalMem: " << std::fixed << std::setprecision(2)
            << bytesToGB(props.totalGlobalMem) << " GB" << std::endl;
#if !defined(__CUDACC__)
  std::cout << std::setw(w1) << "maxSharedMemoryPerMultiProcessor: " << std::fixed << std::setprecision(2)
            << bytesToconfig(props.sharedMemPerMultiprocessor) << " config" << std::endl;
#endif
#if defined(__HIPCC__)
  std::cout << std::setw(w1) << "maxSharedMemoryPerMultiProcessor: " << std::fixed << std::setprecision(2)
            << bytesToconfig(props.maxSharedMemoryPerMultiProcessor) << " config" << std::endl;
#endif
  std::cout << std::setw(w1) << "totalConstMem: " << props.totalConstMem << std::endl;
  std::cout << std::setw(w1) << "sharedMemPerBlock: " << (float)props.sharedMemPerBlock / 1024.0 << " config"
            << std::endl;
  std::cout << std::setw(w1) << "canMapHostMemory: " << props.canMapHostMemory << std::endl;
  std::cout << std::setw(w1) << "regsPerBlock: " << props.regsPerBlock << std::endl;
  std::cout << std::setw(w1) << "warpSize: " << props.warpSize << std::endl;
  std::cout << std::setw(w1) << "l2CacheSize: " << props.l2CacheSize << std::endl;
  std::cout << std::setw(w1) << "computeMode: " << props.computeMode << std::endl;
  std::cout << std::setw(w1) << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << std::endl;
  std::cout << std::setw(w1) << "maxThreadsDim.x: " << props.maxThreadsDim[0] << std::endl;
  std::cout << std::setw(w1) << "maxThreadsDim.y: " << props.maxThreadsDim[1] << std::endl;
  std::cout << std::setw(w1) << "maxThreadsDim.z: " << props.maxThreadsDim[2] << std::endl;
  std::cout << std::setw(w1) << "maxGridSize.x: " << props.maxGridSize[0] << std::endl;
  std::cout << std::setw(w1) << "maxGridSize.y: " << props.maxGridSize[1] << std::endl;
  std::cout << std::setw(w1) << "maxGridSize.z: " << props.maxGridSize[2] << std::endl;
  std::cout << std::setw(w1) << "major: " << props.major << std::endl;
  std::cout << std::setw(w1) << "minor: " << props.minor << std::endl;
  std::cout << std::setw(w1) << "concurrentKernels: " << props.concurrentKernels << std::endl;
  std::cout << std::setw(w1) << "cooperativeLaunch: " << props.cooperativeLaunch << std::endl;
  std::cout << std::setw(w1) << "cooperativeMultiDeviceLaunch: " << props.cooperativeMultiDeviceLaunch << std::endl;
#if defined(__HIPCC__)
  std::cout << std::setw(w1) << "arch.hasGlobalInt32Atomics: " << props.arch.hasGlobalInt32Atomics << std::endl;
  std::cout << std::setw(w1) << "arch.hasGlobalFloatAtomicExch: " << props.arch.hasGlobalFloatAtomicExch
            << std::endl;
  std::cout << std::setw(w1) << "arch.hasSharedInt32Atomics: " << props.arch.hasSharedInt32Atomics << std::endl;
  std::cout << std::setw(w1) << "arch.hasSharedFloatAtomicExch: " << props.arch.hasSharedFloatAtomicExch
            << std::endl;
  std::cout << std::setw(w1) << "arch.hasFloatAtomicAdd: " << props.arch.hasFloatAtomicAdd << std::endl;
  std::cout << std::setw(w1) << "arch.hasGlobalInt64Atomics: " << props.arch.hasGlobalInt64Atomics << std::endl;
  std::cout << std::setw(w1) << "arch.hasSharedInt64Atomics: " << props.arch.hasSharedInt64Atomics << std::endl;
  std::cout << std::setw(w1) << "arch.hasDoubles: " << props.arch.hasDoubles << std::endl;
  std::cout << std::setw(w1) << "arch.hasWarpVote: " << props.arch.hasWarpVote << std::endl;
  std::cout << std::setw(w1) << "arch.hasWarpBallot: " << props.arch.hasWarpBallot << std::endl;
  std::cout << std::setw(w1) << "arch.hasWarpShuffle: " << props.arch.hasWarpShuffle << std::endl;
  std::cout << std::setw(w1) << "arch.hasFunnelShift: " << props.arch.hasFunnelShift << std::endl;
  std::cout << std::setw(w1) << "arch.hasThreadFenceSystem: " << props.arch.hasThreadFenceSystem << std::endl;
  std::cout << std::setw(w1) << "arch.hasSyncThreadsExt: " << props.arch.hasSyncThreadsExt << std::endl;
  std::cout << std::setw(w1) << "arch.hasSurfaceFuncs: " << props.arch.hasSurfaceFuncs << std::endl;
  std::cout << std::setw(w1) << "arch.has3dGrid: " << props.arch.has3dGrid << std::endl;
  std::cout << std::setw(w1) << "arch.hasDynamicParallelism: " << props.arch.hasDynamicParallelism << std::endl;
  std::cout << std::setw(w1) << "gcnArchName: " << props.gcnArchName << std::endl;
#endif
  std::cout << std::setw(w1) << "isIntegrated: " << props.integrated << std::endl;
  std::cout << std::setw(w1) << "maxTexture1D: " << props.maxTexture1D << std::endl;
  std::cout << std::setw(w1) << "maxTexture2D.width: " << props.maxTexture2D[0] << std::endl;
  std::cout << std::setw(w1) << "maxTexture2D.height: " << props.maxTexture2D[1] << std::endl;
  std::cout << std::setw(w1) << "maxTexture3D.width: " << props.maxTexture3D[0] << std::endl;
  std::cout << std::setw(w1) << "maxTexture3D.height: " << props.maxTexture3D[1] << std::endl;
  std::cout << std::setw(w1) << "maxTexture3D.depth: " << props.maxTexture3D[2] << std::endl;
#if defined(__HIPCC__)
  std::cout << std::setw(w1) << "isLargeBar: " << props.isLargeBar << std::endl;
  std::cout << std::setw(w1) << "asicRevision: " << props.asicRevision << std::endl;
#endif

  int deviceCnt;
  GPUCHECK(cudaGetDeviceCount(&deviceCnt));
  std::cout << std::setw(w1) << "peers: ";
  for (int i = 0; i < deviceCnt; i++) {
    int isPeer;
    GPUCHECK(cudaDeviceCanAccessPeer(&isPeer, i, deviceId));
    if (isPeer) {
      std::cout << "device#" << i << " ";
    }
  }
  std::cout << std::endl;
  std::cout << std::setw(w1) << "non-peers: ";
  for (int i = 0; i < deviceCnt; i++) {
    int isPeer;
    GPUCHECK(cudaDeviceCanAccessPeer(&isPeer, i, deviceId));
    if (!isPeer) {
      std::cout << "device#" << i << " ";
    }
  }
  std::cout << std::endl;

  size_t free, total;
  GPUCHECK(cudaMemGetInfo(&free, &total));

  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::setw(w1) << "memInfo.total: " << bytesToGB(total) << " GB" << std::endl;
  std::cout << std::setw(w1) << "memInfo.free:  " << bytesToGB(free) << " GB (" << std::setprecision(0)
            << (float)free / total * 100.0 << "%)" << std::endl;
}

template <class chunk_t>
template <typename... T>
float GPUbenchmark<chunk_t>::runSequential(void (*kernel)(chunk_t*, T...),
                                           int nLaunches,
                                           int chunkId,
                                           int nBlocks,
                                           int nThreads,
                                           T&... args) // run for each chunk
{
  cudaEvent_t start, stop;
  cudaStream_t stream;
  GPUCHECK(cudaStreamCreate(&stream));

  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  chunk_t* chunkPtr = getPartPtr<chunk_t>(mState.scratchPtr, mState.chunkReservedGB, chunkId);

  // Warm up
  (*kernel)<<<nBlocks, nThreads, 0, stream>>>(chunkPtr, args...);

  GPUCHECK(cudaEventCreate(&start));
  GPUCHECK(cudaEventCreate(&stop));

  GPUCHECK(cudaEventRecord(start));
  for (auto iLaunch{0}; iLaunch < nLaunches; ++iLaunch) {           // Schedule all the requested kernel launches
    (*kernel)<<<nBlocks, nThreads, 0, stream>>>(chunkPtr, args...); // NOLINT: clang-tidy false-positive
  }
  GPUCHECK(cudaEventRecord(stop)); // record checkpoint

  GPUCHECK(cudaEventSynchronize(stop)); // synchronize executions
  float milliseconds{0.f};
  GPUCHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  GPUCHECK(cudaEventDestroy(start));
  GPUCHECK(cudaEventDestroy(stop));

  GPUCHECK(cudaStreamDestroy(stream));
  return milliseconds;
}

template <class chunk_t>
template <typename... T>
std::vector<float> GPUbenchmark<chunk_t>::runConcurrent(void (*kernel)(chunk_t*, T...),
                                                        int nChunks,
                                                        int nLaunches,
                                                        int dimStreams,
                                                        int nBlocks,
                                                        int nThreads,
                                                        T&... args)
{
  std::vector<cudaEvent_t> starts(nChunks), stops(nChunks);
  std::vector<cudaStream_t> streams(dimStreams);

  std::vector<float> results(nChunks);
  GPUCHECK(cudaSetDevice(mOptions.deviceId));

  for (auto iStream{0}; iStream < dimStreams; ++iStream) {
    GPUCHECK(cudaStreamCreate(&(streams.at(iStream)))); // round-robin on stream pool
  }

  for (auto iChunk{0}; iChunk < nChunks; ++iChunk) {
    GPUCHECK(cudaEventCreate(&(starts[iChunk])));
    GPUCHECK(cudaEventCreate(&(stops[iChunk])));
  }

  // Warm up on every chunk
  for (auto iChunk{0}; iChunk < nChunks; ++iChunk) {
    chunk_t* chunkPtr = getPartPtr<chunk_t>(mState.scratchPtr, mState.chunkReservedGB, iChunk);
    (*kernel)<<<nBlocks, nThreads, 0, streams[iChunk % dimStreams]>>>(chunkPtr, args...);
  }

  for (auto iChunk{0}; iChunk < nChunks; ++iChunk) {
    chunk_t* chunkPtr = getPartPtr<chunk_t>(mState.scratchPtr, mState.chunkReservedGB, iChunk);
    GPUCHECK(cudaEventRecord(starts[iChunk], streams[iChunk % dimStreams]));
    for (auto iLaunch{0}; iLaunch < nLaunches; ++iLaunch) {
      (*kernel)<<<nBlocks, nThreads, 0, streams[iChunk % dimStreams]>>>(chunkPtr, args...);
    }
    GPUCHECK(cudaEventRecord(stops[iChunk], streams[iChunk % dimStreams]));
  }

  for (auto iChunk{0}; iChunk < nChunks; ++iChunk) {
    GPUCHECK(cudaEventSynchronize(stops[iChunk]));
    GPUCHECK(cudaEventElapsedTime(&(results.at(iChunk)), starts[iChunk], stops[iChunk]));
    GPUCHECK(cudaEventDestroy(starts[iChunk]));
    GPUCHECK(cudaEventDestroy(stops[iChunk]));
  }

  for (auto iStream{0}; iStream < dimStreams; ++iStream) {
    GPUCHECK(cudaStreamDestroy(streams[iStream]));
  }

  return results;
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::printDevices()
{
  int deviceCnt;
  GPUCHECK(cudaGetDeviceCount(&deviceCnt));

  for (int i = 0; i < deviceCnt; i++) {
    GPUCHECK(cudaSetDevice(i));
    printDeviceProp(i);
  }
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::globalInit()
{
  cudaDeviceProp props;
  size_t free;

  // Fetch and store features
  GPUCHECK(cudaGetDeviceProperties(&props, mOptions.deviceId));
  GPUCHECK(cudaMemGetInfo(&free, &mState.totalMemory));
  GPUCHECK(cudaSetDevice(mOptions.deviceId));

  mState.chunkReservedGB = mOptions.chunkReservedGB;
  mState.iterations = mOptions.kernelLaunches;
  mState.streams = mOptions.streams;
  mState.nMultiprocessors = props.multiProcessorCount;
  mState.nMaxThreadsPerBlock = props.maxThreadsPerMultiProcessor;
  mState.nMaxThreadsPerDimension = props.maxThreadsDim[0];
  mState.scratchSize = static_cast<long int>(mOptions.freeMemoryFractionToAllocate * free);
  std::cout << " ◈ Running on: \033[1;31m" << props.name << "\e[0m" << std::endl;

  // Allocate scratch on GPU
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&mState.scratchPtr), mState.scratchSize));

  mState.computeScratchPtrs();
  GPUCHECK(cudaMemset(mState.scratchPtr, 0, mState.scratchSize))

  std::cout << "   ├ Buffer type: \e[1m" << getType<chunk_t>() << "\e[0m" << std::endl
            << "   ├ Allocated: " << std::setprecision(2) << bytesToGB(mState.scratchSize) << "/" << std::setprecision(2) << bytesToGB(mState.totalMemory)
            << "(GB) [" << std::setprecision(3) << (100.f) * (mState.scratchSize / (float)mState.totalMemory) << "%]\n"
            << "   ├ Number of streams allocated: " << mState.getStreamsPoolSize() << "\n"
            << "   ├ Number of scratch chunks: " << mState.getMaxChunks() << " of " << mOptions.chunkReservedGB << " GB each\n"
            << "   └ Each chunk can store up to: " << mState.getChunkCapacity() << " elements" << std::endl
            << std::endl;
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::initTest(Test test)
{
  std::cout << " ◈ \033[1;33m" << getType<chunk_t>() << "\033[0m " << test << " benchmark with \e[1m" << mOptions.nTests << "\e[0m runs and \e[1m" << mOptions.kernelLaunches << "\e[0m kernel launches" << std::endl;
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::runTest(Test test, Mode mode, KernelConfig config)
{
  mResultWriter.get()->addBenchmarkEntry(getTestName(mode, test, config), getType<chunk_t>(), mState.getMaxChunks());
  auto dimGrid{mState.nMultiprocessors};
  auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock) * mOptions.threadPoolFraction};
  auto nBlocks{(config == KernelConfig::Single) ? 1 : (config == KernelConfig::Multi) ? dimGrid / mState.getMaxChunks()
                                                                                      : dimGrid};
  auto chunks{mState.getMaxChunks()};
  auto capacity{mState.getChunkCapacity()};
  void (*kernel)(chunk_t*, size_t);

  switch (test) {
    case Test::Read: {
      kernel = &gpu::read_k<chunk_t>;
      break;
    }
    case Test::Write: {
      kernel = &gpu::write_k<chunk_t>;
      break;
    }
    case Test::Copy: {
      kernel = &gpu::copy_k<chunk_t>;
      break;
    }
  }

  for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
    std::cout << "   ├ " << mode << " " << test << " " << config << " block(s) (" << measurement + 1 << "/" << mOptions.nTests << "): \n"
              << "   │   - blocks per kernel: " << nBlocks << "/" << dimGrid << "\n"
              << "   │   - threads per block: " << nThreads << "\n"
              << "   │   - per chunk throughput:\n";
    if (mode == Mode::Sequential) {
      for (auto iChunk{0}; iChunk < mState.getMaxChunks(); ++iChunk) { // loop over single chunks separately
        auto result = runSequential(kernel,
                                    mState.getNKernelLaunches(),
                                    iChunk,
                                    nBlocks,
                                    nThreads, // args...
                                    capacity);
        auto throughput = computeThroughput(test, result, mState.chunkReservedGB, mState.getNKernelLaunches());
        std::cout << "   │     " << ((mState.getMaxChunks() - iChunk != 1) ? "├ " : "└ ") << iChunk + 1 << "/" << mState.getMaxChunks() << ": \e[1m" << throughput << " GB/s \e[0m (" << result * 1e-3 << " s)\n";
        mResultWriter.get()->storeBenchmarkEntry(test, iChunk, result, mState.chunkReservedGB, mState.getNKernelLaunches());
      }
    } else {
      auto results = runConcurrent(kernel,
                                   mState.getMaxChunks(), // nStreams
                                   mState.getNKernelLaunches(),
                                   mState.getStreamsPoolSize(),
                                   nBlocks,
                                   nThreads,
                                   capacity);
      for (auto iChunk{0}; iChunk < results.size(); ++iChunk) {
        auto throughput = computeThroughput(test, results[iChunk], mState.chunkReservedGB, mState.getNKernelLaunches());
        std::cout << "   │     " << ((results.size() - iChunk != 1) ? "├ " : "└ ") << iChunk + 1 << "/" << results.size() << ": \e[1m" << throughput << " GB/s \e[0m (" << results[iChunk] * 1e-3 << " s)\n";
        mResultWriter.get()->storeBenchmarkEntry(test, iChunk, results[iChunk], mState.chunkReservedGB, mState.getNKernelLaunches());
      }
    }
    mResultWriter.get()->snapshotBenchmark();
  }
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::finalizeTest(Test test)
{
  std::cout << "   └\033[1;32m done\033[0m" << std::endl;
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::globalFinalize()
{
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaFree(mState.scratchPtr));
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::run()
{
  globalInit();

  for (auto& test : mOptions.tests) {
    initTest(test);
    for (auto& mode : mOptions.modes) {
      for (auto& config : mOptions.pools) {
        runTest(test, mode, config);
      }
    }
    finalizeTest(test);
  }

  globalFinalize();
}

template class GPUbenchmark<char>;
template class GPUbenchmark<size_t>;
template class GPUbenchmark<int>;
template class GPUbenchmark<int4>;

} // namespace benchmark
} // namespace o2