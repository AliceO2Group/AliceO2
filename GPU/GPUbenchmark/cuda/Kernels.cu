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
#include <chrono>
#include <cstdio>
#include <numeric>

// Memory partitioning schema
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

bool checkTestChunks(std::vector<std::pair<float, float>>& chunks, size_t availMemSizeGB)
{
  if (!chunks.size()) {
    return true;
  }

  bool check{false};

  sort(chunks.begin(), chunks.end());
  for (size_t iChunk{0}; iChunk < chunks.size(); ++iChunk) { // Check boundaries
    if (chunks[iChunk].first + chunks[iChunk].second > availMemSizeGB) {
      check = false;
      break;
    }
    if (iChunk > 0) { // Check intersections
      if (chunks[iChunk].first < chunks[iChunk - 1].first + chunks[iChunk - 1].second) {
        check = false;
        break;
      }
    }
    check = true;
  }
  return check;
}

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
  size_t chunkSize,
  int prime)
{
  chunk_t sink{0};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    sink += chunkPtr[(i * prime) % chunkSize];
  }
  chunkPtr[threadIdx.x] = sink;
}

// Random write
template <class chunk_t>
__global__ void rand_write_k(
  chunk_t* chunkPtr,
  size_t chunkSize,
  int prime)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    chunkPtr[(i * prime) % chunkSize] = 0;
  }
}

template <>
__global__ void rand_write_k(
  int4* chunkPtr,
  size_t chunkSize,
  int prime)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    chunkPtr[(i * prime) % chunkSize] = {0, 1, 0, 0};
  };
}

// Random copy
template <class chunk_t>
__global__ void rand_copy_k(
  chunk_t* chunkPtr,
  size_t chunkSize,
  int prime)
{
  size_t offset = chunkSize / 2;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < offset; i += blockDim.x * gridDim.x) {
    chunkPtr[(i * prime) % offset] = chunkPtr[offset + (i * prime) % offset]; // might be % = 0...
  }
}

// Distributed read
template <class chunk_t>
__global__ void read_dist_k(
  chunk_t** block_ptr,
  size_t* block_size)
{
  chunk_t sink{0};
  chunk_t* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    sink += ptr[i];
  }
  ptr[threadIdx.x] = sink;
}

// Distributed write
template <class chunk_t>
__global__ void write_dist_k(
  chunk_t** block_ptr,
  size_t* block_size)
{
  chunk_t* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    ptr[i] = 0;
  }
}

template <>
__global__ void write_dist_k(
  int4** block_ptr,
  size_t* block_size)
{
  int4* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    ptr[i] = {0, 1, 0, 0};
  }
}

// Distributed copy
template <class chunk_t>
__global__ void copy_dist_k(
  chunk_t** block_ptr,
  size_t* block_size)
{
  chunk_t* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  size_t offset = n / 2;
  for (size_t i = threadIdx.x; i < offset; i += blockDim.x) {
    ptr[i] = ptr[offset + i];
  }
}

// Distributed Random read
template <class chunk_t>
__global__ void rand_read_dist_k(
  chunk_t** block_ptr,
  size_t* block_size,
  int prime)
{
  chunk_t sink{0};
  chunk_t* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    sink += ptr[(i * prime) % n];
  }
  ptr[threadIdx.x] = sink;
}

// Distributed Random write
template <class chunk_t>
__global__ void rand_write_dist_k(
  chunk_t** block_ptr,
  size_t* block_size,
  int prime)
{
  chunk_t* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    ptr[(i * prime) % n] = 0;
  }
}

template <>
__global__ void rand_write_dist_k(
  int4** block_ptr,
  size_t* block_size,
  int prime)
{
  int4* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    ptr[(i * prime) % n] = {0, 1, 0, 0};
  }
}

// Distributed Random copy
template <class chunk_t>
__global__ void rand_copy_dist_k(
  chunk_t** block_ptr,
  size_t* block_size,
  int prime)
{
  chunk_t* ptr = block_ptr[blockIdx.x];
  size_t n = block_size[blockIdx.x];
  size_t offset = n / 2;
  for (size_t i = threadIdx.x; i < offset; i += blockDim.x) {
    ptr[(i * prime) % offset] = ptr[offset + (i * prime) % offset];
  }
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
float GPUbenchmark<chunk_t>::runSequential(void (*kernel)(chunk_t*, size_t, T...),
                                           std::pair<float, float>& chunk,
                                           int nLaunches,
                                           int nBlocks,
                                           int nThreads,
                                           T&... args) // run for each chunk
{
  float milliseconds{0.f};
  cudaEvent_t start, stop;
  cudaStream_t stream;
  GPUCHECK(cudaStreamCreate(&stream));
  GPUCHECK(cudaSetDevice(mOptions.deviceId));

  chunk_t* chunkPtr = getCustomPtr<chunk_t>(mState.scratchPtr, chunk.first);

  // Warm up
  (*kernel)<<<nBlocks, nThreads, 0, stream>>>(chunkPtr, getBufferCapacity<chunk_t>(chunk.second, mOptions.prime), args...);
  GPUCHECK(cudaGetLastError());
  GPUCHECK(cudaEventCreate(&start));
  GPUCHECK(cudaEventCreate(&stop));

  GPUCHECK(cudaEventRecord(start));
  for (auto iLaunch{0}; iLaunch < nLaunches; ++iLaunch) {                                                                     // Schedule all the requested kernel launches
    (*kernel)<<<nBlocks, nThreads, 0, stream>>>(chunkPtr, getBufferCapacity<chunk_t>(chunk.second, mOptions.prime), args...); // NOLINT: clang-tidy false-positive
    GPUCHECK(cudaGetLastError());
  }
  GPUCHECK(cudaEventRecord(stop));      // record checkpoint
  GPUCHECK(cudaEventSynchronize(stop)); // synchronize executions
  GPUCHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  GPUCHECK(cudaEventDestroy(start));
  GPUCHECK(cudaEventDestroy(stop));
  GPUCHECK(cudaStreamDestroy(stream));

  return milliseconds;
}

template <class chunk_t>
template <typename... T>
std::vector<float> GPUbenchmark<chunk_t>::runConcurrent(void (*kernel)(chunk_t*, size_t, T...),
                                                        std::vector<std::pair<float, float>>& chunkRanges,
                                                        int nLaunches,
                                                        int dimStreams,
                                                        int nBlocks,
                                                        int nThreads,
                                                        T&... args)
{
  auto nChunks = chunkRanges.size();
  std::vector<float> results(nChunks + 1); // last spot is for the host time
  std::vector<cudaEvent_t> starts(nChunks), stops(nChunks);
  std::vector<cudaStream_t> streams(dimStreams);

  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  for (auto iStream{0}; iStream < dimStreams; ++iStream) {
    GPUCHECK(cudaStreamCreate(&(streams.at(iStream)))); // round-robin on stream pool
  }
  for (size_t iChunk{0}; iChunk < nChunks; ++iChunk) {
    GPUCHECK(cudaEventCreate(&(starts[iChunk])));
    GPUCHECK(cudaEventCreate(&(stops[iChunk])));
  }

  // Warm up on every chunk
  for (size_t iChunk{0}; iChunk < nChunks; ++iChunk) {
    auto& chunk = chunkRanges[iChunk];
    chunk_t* chunkPtr = getCustomPtr<chunk_t>(mState.scratchPtr, chunk.first);
    (*kernel)<<<nBlocks, nThreads, 0, streams[iChunk % dimStreams]>>>(chunkPtr, getBufferCapacity<chunk_t>(chunk.second, mOptions.prime), args...);
  }
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t iChunk{0}; iChunk < nChunks; ++iChunk) {
    auto& chunk = chunkRanges[iChunk];
    chunk_t* chunkPtr = getCustomPtr<chunk_t>(mState.scratchPtr, chunk.first);
    GPUCHECK(cudaEventRecord(starts[iChunk], streams[iChunk % dimStreams]));
    for (auto iLaunch{0}; iLaunch < nLaunches; ++iLaunch) {
      (*kernel)<<<nBlocks, nThreads, 0, streams[iChunk % dimStreams]>>>(chunkPtr, getBufferCapacity<chunk_t>(chunk.second, mOptions.prime), args...);
    }
    GPUCHECK(cudaEventRecord(stops[iChunk], streams[iChunk % dimStreams]));
  }

  for (size_t iChunk{0}; iChunk < nChunks; ++iChunk) {
    GPUCHECK(cudaEventSynchronize(stops[iChunk]));
    GPUCHECK(cudaEventElapsedTime(&(results.at(iChunk)), starts[iChunk], stops[iChunk]));
    GPUCHECK(cudaEventDestroy(starts[iChunk]));
    GPUCHECK(cudaEventDestroy(stops[iChunk]));
  }
  GPUCHECK(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff_t{end - start};

  for (auto iStream{0}; iStream < dimStreams; ++iStream) {
    GPUCHECK(cudaStreamDestroy(streams[iStream]));
  }

  results[nChunks] = diff_t.count(); // register host time on latest spot
  return results;
}

template <class chunk_t>
template <typename... T>
float GPUbenchmark<chunk_t>::runDistributed(void (*kernel)(chunk_t**, size_t*, T...),
                                            std::vector<std::pair<float, float>>& chunkRanges,
                                            int nLaunches,
                                            size_t nBlocks,
                                            int nThreads,
                                            T&... args)
{
  std::vector<chunk_t*> chunkPtrs(chunkRanges.size()); // Pointers to the beginning of each chunk
  std::vector<chunk_t*> ptrPerBlocks(nBlocks);         // Pointers for each block
  std::vector<size_t> perBlockCapacity(nBlocks);       // Capacity of sub-buffer for block

  float totChunkGB{0.f};
  size_t totComputedBlocks{0};

  for (size_t iChunk{0}; iChunk < chunkRanges.size(); ++iChunk) {
    chunkPtrs[iChunk] = getCustomPtr<chunk_t>(mState.scratchPtr, chunkRanges[iChunk].first);
    totChunkGB += chunkRanges[iChunk].second;
  }
  int index{0};
  for (size_t iChunk{0}; iChunk < chunkRanges.size(); ++iChunk) {
    float percFromMem = chunkRanges[iChunk].second / totChunkGB;
    int blocksPerChunk = percFromMem * nBlocks;
    totComputedBlocks += blocksPerChunk;
    for (int iBlock{0}; iBlock < blocksPerChunk; ++iBlock, ++index) {
      float memPerBlock = chunkRanges[iChunk].second / blocksPerChunk;
      ptrPerBlocks[index] = getCustomPtr<chunk_t>(chunkPtrs[iChunk], iBlock * memPerBlock);
      perBlockCapacity[index] = getBufferCapacity<chunk_t>(memPerBlock, mOptions.prime);
    }
  }

  if (totComputedBlocks != nBlocks) {
    std::cerr << "   │   - \033[1;33mWarning: Sum of used blocks (" << totComputedBlocks
              << ") is different from requested one (" << nBlocks << ")!\e[0m"
              << std::endl;
  }

  if (mOptions.dumpChunks) {
    for (size_t iChunk{0}; iChunk < totComputedBlocks; ++iChunk) {
      std::cout << "   │   - block " << iChunk << " address: " << ptrPerBlocks[iChunk] << ", size: " << perBlockCapacity[iChunk] << std::endl;
    }
  }

  // Setup
  chunk_t** block_ptr;
  size_t* block_size;
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&block_ptr), nBlocks * sizeof(chunk_t*)));
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&block_size), nBlocks * sizeof(size_t)));
  GPUCHECK(cudaMemcpy(block_ptr, ptrPerBlocks.data(), nBlocks * sizeof(chunk_t*), cudaMemcpyHostToDevice));
  GPUCHECK(cudaMemcpy(block_size, perBlockCapacity.data(), nBlocks * sizeof(size_t), cudaMemcpyHostToDevice));

  float milliseconds{0.f};
  cudaEvent_t start, stop;
  cudaStream_t stream;

  GPUCHECK(cudaStreamCreate(&stream));
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaEventCreate(&start));
  GPUCHECK(cudaEventCreate(&stop));

  // Warm up
  (*kernel)<<<totComputedBlocks, nThreads, 0, stream>>>(block_ptr, block_size, args...);

  GPUCHECK(cudaEventRecord(start));
  for (auto iLaunch{0}; iLaunch < nLaunches; ++iLaunch) {                                  // Schedule all the requested kernel launches
    (*kernel)<<<totComputedBlocks, nThreads, 0, stream>>>(block_ptr, block_size, args...); // NOLINT: clang-tidy false-positive
  }
  GPUCHECK(cudaEventRecord(stop));      // record checkpoint
  GPUCHECK(cudaEventSynchronize(stop)); // synchronize executions
  GPUCHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  GPUCHECK(cudaEventDestroy(start));
  GPUCHECK(cudaEventDestroy(stop));
  GPUCHECK(cudaStreamDestroy(stream));
  return milliseconds;
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
  mState.testChunks = mOptions.testChunks;
  if (!checkTestChunks(mState.testChunks, mOptions.freeMemoryFractionToAllocate * free / GB)) {
    std::cerr << "Failed to configure memory chunks: check arbitrary chunks boundaries." << std::endl;
    exit(1);
  }
  mState.nMultiprocessors = props.multiProcessorCount;
  mState.nMaxThreadsPerBlock = props.maxThreadsPerMultiProcessor;
  mState.nMaxThreadsPerDimension = props.maxThreadsDim[0];
  mState.scratchSize = static_cast<long int>(mOptions.freeMemoryFractionToAllocate * free);

  if (mState.testChunks.empty()) {
    for (auto j{0}; j < mState.getMaxChunks() * mState.chunkReservedGB; j += mState.chunkReservedGB) {
      mState.testChunks.emplace_back(j, mState.chunkReservedGB);
    }
  }

  if (!mOptions.raw) {
    std::cout << " ◈ Running on: \033[1;31m" << props.name << "\e[0m" << std::endl;
  }
  // Allocate scratch on GPU
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&mState.scratchPtr), mState.scratchSize));
  GPUCHECK(cudaMemset(mState.scratchPtr, 0, mState.scratchSize))

  if (!mOptions.raw) {
    std::cout << "   ├ Buffer type: \e[1m" << getType<chunk_t>() << "\e[0m" << std::endl
              << "   ├ Allocated: " << std::setprecision(2) << bytesToGB(mState.scratchSize) << "/" << std::setprecision(2) << bytesToGB(mState.totalMemory)
              << "(GB) [" << std::setprecision(3) << (100.f) * (mState.scratchSize / (float)mState.totalMemory) << "%]\n"
              << "   └ Available streams: " << mState.getStreamsPoolSize() << "\n\n";
  }
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::initTest(Test test)
{
  if (!mOptions.raw) {
    std::cout << " ◈ \033[1;33m" << getType<chunk_t>() << "\033[0m " << test << " benchmark with \e[1m" << mOptions.nTests << "\e[0m runs and \e[1m" << mOptions.kernelLaunches << "\e[0m kernel launches" << std::endl;
  }
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::runTest(Test test, Mode mode, KernelConfig config)
{
  auto dimGrid{mState.nMultiprocessors};
  auto nBlocks{(config == KernelConfig::Single) ? 1 : (config == KernelConfig::Multi) ? dimGrid / mState.testChunks.size()
                                                    : (config == KernelConfig::All)   ? dimGrid
                                                                                      : mOptions.numBlocks};
  size_t nThreads;
  if (mOptions.numThreads < 0) {
    nThreads = std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock);
  } else {
    nThreads = mOptions.numThreads;
  }
  nThreads *= mOptions.threadPoolFraction;

  void (*kernel)(chunk_t*, size_t) = &gpu::read_k<chunk_t>;                                   // Initialising to a default value
  void (*kernel_distributed)(chunk_t**, size_t*) = &gpu::read_dist_k<chunk_t>;                // Initialising to a default value
  void (*kernel_rand)(chunk_t*, size_t, int) = &gpu::rand_read_k<chunk_t>;                    // Initialising to a default value
  void (*kernel_rand_distributed)(chunk_t**, size_t*, int) = &gpu::rand_read_dist_k<chunk_t>; // Initialising to a default value

  bool is_random{false};

  if (mode != Mode::Distributed) {
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
      case Test::RandomRead: {
        kernel_rand = &gpu::rand_read_k<chunk_t>;
        is_random = true;
        break;
      }
      case Test::RandomWrite: {
        kernel_rand = &gpu::rand_write_k<chunk_t>;
        is_random = true;
        break;
      }
      case Test::RandomCopy: {
        kernel_rand = &gpu::rand_copy_k<chunk_t>;
        is_random = true;
        break;
      }
    }
  } else {
    switch (test) {
      case Test::Read: {
        kernel_distributed = &gpu::read_dist_k<chunk_t>;
        break;
      }
      case Test::Write: {
        kernel_distributed = &gpu::write_dist_k<chunk_t>;
        break;
      }
      case Test::Copy: {
        kernel_distributed = &gpu::copy_dist_k<chunk_t>;
        break;
      }
      case Test::RandomRead: {
        kernel_rand_distributed = &gpu::rand_read_dist_k<chunk_t>;
        is_random = true;
        break;
      }
      case Test::RandomWrite: {
        kernel_rand_distributed = &gpu::rand_write_dist_k<chunk_t>;
        is_random = true;
        break;
      }
      case Test::RandomCopy: {
        kernel_rand_distributed = &gpu::rand_copy_dist_k<chunk_t>;
        is_random = true;
        break;
      }
    }
  }

  for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
    if (!mOptions.raw) {
      std::cout << "   ├ " << mode << " " << test << " " << config << " block(s) (" << measurement + 1 << "/" << mOptions.nTests << "): \n"
                << "   │   - blocks per kernel: " << nBlocks << "/" << dimGrid << "\n"
                << "   │   - threads per block: " << (int)nThreads << "\n";
    }
    if (mode == Mode::Sequential) {
      if (!mOptions.raw) {
        std::cout << "   │   - per chunk throughput:\n";
      }
      for (size_t iChunk{0}; iChunk < mState.testChunks.size(); ++iChunk) { // loop over single chunks separately
        auto& chunk = mState.testChunks[iChunk];
        float result{0.f};
        if (!is_random) {
          result = runSequential(kernel,
                                 chunk,
                                 mState.getNKernelLaunches(),
                                 nBlocks,
                                 nThreads);
        } else {
          result = runSequential(kernel_rand,
                                 chunk,
                                 mState.getNKernelLaunches(),
                                 nBlocks,
                                 nThreads,
                                 mOptions.prime);
        }
        float chunkSize = (float)getBufferCapacity<chunk_t>(chunk.second, mOptions.prime) * sizeof(chunk_t) / (float)GB;
        auto throughput = computeThroughput(test, result, chunkSize, mState.getNKernelLaunches());
        if (!mOptions.raw) {
          std::cout << "   │     " << ((mState.testChunks.size() - iChunk != 1) ? "├ " : "└ ") << iChunk + 1 << "/" << mState.testChunks.size()
                    << ": [" << chunk.first << "-" << chunk.first + chunk.second << ") \e[1m" << throughput << " GB/s \e[0m(" << result * 1e-3 << " s)\n";
        } else {
          std::cout << "" << measurement << "\t" << iChunk << "\t" << throughput << "\t" << chunkSize << "\t" << result << std::endl;
        }
      }
    } else if (mode == Mode::Concurrent) {
      if (!mOptions.raw) {
        std::cout << "   │   - per chunk throughput:\n";
      }
      std::vector<float> results;
      if (!is_random) {
        results = runConcurrent(kernel,
                                mState.testChunks,
                                mState.getNKernelLaunches(),
                                mState.getStreamsPoolSize(),
                                nBlocks,
                                nThreads);
      } else {
        results = runConcurrent(kernel_rand,
                                mState.testChunks,
                                mState.getNKernelLaunches(),
                                mState.getStreamsPoolSize(),
                                nBlocks,
                                nThreads,
                                mOptions.prime);
      }
      float sum{0};
      for (size_t iChunk{0}; iChunk < mState.testChunks.size(); ++iChunk) {
        auto& chunk = mState.testChunks[iChunk];
        float chunkSize = (float)getBufferCapacity<chunk_t>(chunk.second, mOptions.prime) * sizeof(chunk_t) / (float)GB;
        auto throughput = computeThroughput(test, results[iChunk], chunkSize, mState.getNKernelLaunches());
        sum += throughput;
        if (!mOptions.raw) {
          std::cout << "   │     " << ((mState.testChunks.size() - iChunk != 1) ? "├ " : "└ ") << iChunk + 1 << "/" << mState.testChunks.size()
                    << ": [" << chunk.first << "-" << chunk.first + chunk.second << ") \e[1m" << throughput << " GB/s \e[0m(" << results[iChunk] * 1e-3 << " s)\n";
        } else {
          std::cout << "" << measurement << "\t" << iChunk << "\t" << throughput << "\t" << chunkSize << "\t" << results[iChunk] << std::endl;
        }
      }
      if (mState.testChunks.size() > 1) {
        if (!mOptions.raw) {
          std::cout << "   │   - total throughput: \e[1m" << sum << " GB/s \e[0m" << std::endl;
        }
      }

      // Add throughput computed via system time measurement
      float tot{0};
      for (auto& chunk : mState.testChunks) {
        tot += chunk.second;
      }

      if (!mOptions.raw) {
        std::cout << "   │   - total throughput with host time: \e[1m" << computeThroughput(test, results[mState.testChunks.size()], tot, mState.getNKernelLaunches())
                  << " GB/s \e[0m (" << std::setw(2) << results[mState.testChunks.size()] / 1000 << " s)" << std::endl;
      }
    } else if (mode == Mode::Distributed) {
      float result{0.f};
      if (!is_random) {
        result = runDistributed(kernel_distributed,
                                mState.testChunks,
                                mState.getNKernelLaunches(),
                                nBlocks,
                                nThreads);
      } else {
        result = runDistributed(kernel_rand_distributed,
                                mState.testChunks,
                                mState.getNKernelLaunches(),
                                nBlocks,
                                nThreads,
                                mOptions.prime);
      }
      float tot{0};
      for (auto& chunk : mState.testChunks) {
        float chunkSize = (float)getBufferCapacity<chunk_t>(chunk.second, mOptions.prime) * sizeof(chunk_t) / (float)GB;
        tot += chunkSize;
      }
      auto throughput = computeThroughput(test, result, tot, mState.getNKernelLaunches());
      if (!mOptions.raw) {
        std::cout << "   │     └ throughput: \e[1m" << throughput << " GB/s \e[0m(" << result * 1e-3 << " s)\n";
      } else {
        std::cout << "" << measurement << "\t" << 0 << "\t" << throughput << "\t" << tot << "\t" << result << std::endl;
      }
    }
  }
}

template <class chunk_t>
void GPUbenchmark<chunk_t>::finalizeTest(Test test)
{
  if (!mOptions.raw) {
    std::cout << "   └\033[1;32m done\033[0m" << std::endl;
  }
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
