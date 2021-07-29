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
#if defined(__HIPCC__)
#include "hip/hip_runtime.h"
#endif
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

double bytesToKB(size_t s) { return (double)s / (1024.0); }
double bytesToGB(size_t s) { return (double)s / GB; }

int getCorrespondingRegionId(int Id, int nChunks, int nRegions = 1)
{
  return Id * nRegions / nChunks;
}

template <class T>
std::string getType()
{
  if (typeid(T).name() == typeid(char).name()) {
    return std::string{"char"};
  }
  if (typeid(T).name() == typeid(size_t).name()) {
    return std::string{"unsigned_long"};
  }
  if (typeid(T).name() == typeid(int).name()) {
    return std::string{"int"};
  }
  if (typeid(T).name() == typeid(int4).name()) {
    return std::string{"int4"};
  }
  return std::string{"unknown"};
}

namespace o2
{
namespace benchmark
{
namespace gpu
{

///////////////////////////
// Device functions go here
template <class chunk_type>
__host__ __device__ inline chunk_type* getPartPtrOnScratch(chunk_type* scratchPtr, float chunkReservedGB, size_t partNumber)
{
  return reinterpret_cast<chunk_type*>(reinterpret_cast<char*>(scratchPtr) + static_cast<size_t>(GB * chunkReservedGB) * partNumber);
}

//////////////////
// Kernels go here
// Reading
template <class chunk_type>
__global__ void readChunkSBKernel(
  int chunkId,
  chunk_type* results,
  chunk_type* scratch,
  size_t chunkSize,
  float chunkReservedGB = 1.f)
{
  if (chunkId == blockIdx.x) { // runs only if blockIdx.x is allowed in given split
    chunk_type sink{0};
    chunk_type* ptr = getPartPtrOnScratch(scratch, chunkReservedGB, chunkId);
    for (size_t i = threadIdx.x; i < chunkSize; i += blockDim.x) {
      sink += ptr[i];
    }
    if (sink == static_cast<chunk_type>(1)) {
      results[chunkId] = sink;
    }
  }
}

template <class chunk_type>
__global__ void readChunkMBKernel(
  int chunkId,
  chunk_type* results,
  chunk_type* scratch,
  size_t chunkSize,
  float chunkReservedGB = 1.f)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    if (getPartPtrOnScratch(scratch, chunkReservedGB, chunkId)[i] == static_cast<chunk_type>(1)) { // actual read operation is performed here
      results[chunkId] += getPartPtrOnScratch(scratch, chunkReservedGB, chunkId)[i];               // this case should never happen and waves should be always in sync
    }
  }
}

// Writing
template <class chunk_type>
__global__ void writeChunkSBKernel(
  int chunkId,
  chunk_type* results,
  chunk_type* scratch,
  size_t chunkSize,
  float chunkReservedGB = 1.f)
{
  if (chunkId == blockIdx.x) { // runs only if blockIdx.x is allowed in given split
    for (size_t i = threadIdx.x; i < chunkSize; i += blockDim.x) {
      getPartPtrOnScratch(scratch, chunkReservedGB, chunkId)[i] = 1;
    }
  }
}

template <class chunk_type>
__global__ void writeChunkMBKernel(
  int chunkId,
  chunk_type* results,
  chunk_type* scratch,
  size_t chunkSize,
  float chunkReservedGB = 1.f)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    getPartPtrOnScratch(scratch, chunkReservedGB, chunkId)[i] = 1;
  }
}

// Copying
template <class chunk_type>
__global__ void copyChunkSBKernel(
  int chunkId,
  chunk_type* inputs,
  chunk_type* scratch,
  size_t chunkSize,
  float chunkReservedGB = 1.f)
{
  if (chunkId == blockIdx.x) { // runs only if blockIdx.x is allowed in given split
    for (size_t i = threadIdx.x; i < chunkSize; i += blockDim.x) {
      getPartPtrOnScratch(scratch, chunkReservedGB, chunkId)[i] = inputs[chunkId];
    }
  }
}

template <class chunk_type>
__global__ void copyChunkMBKernel(
  int chunkId,
  chunk_type* inputs,
  chunk_type* scratch,
  size_t chunkSize,
  float chunkReservedGB = 1.f)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < chunkSize; i += blockDim.x * gridDim.x) {
    getPartPtrOnScratch(scratch, chunkReservedGB, chunkId)[i] = inputs[chunkId];
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
            << bytesToKB(props.sharedMemPerMultiprocessor) << " KB" << std::endl;
#endif
#if defined(__HIPCC__)
  std::cout << std::setw(w1) << "maxSharedMemoryPerMultiProcessor: " << std::fixed << std::setprecision(2)
            << bytesToKB(props.maxSharedMemoryPerMultiProcessor) << " KB" << std::endl;
#endif
  std::cout << std::setw(w1) << "totalConstMem: " << props.totalConstMem << std::endl;
  std::cout << std::setw(w1) << "sharedMemPerBlock: " << (float)props.sharedMemPerBlock / 1024.0 << " KB"
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

template <class chunk_type>
template <typename... T>
float GPUbenchmark<chunk_type>::benchmarkSync(void (*kernel)(T...),
                                              int nLaunches, int blocks, int threads, T&... args) // run for each chunk (id is passed in variadic args)
{
  cudaEvent_t start, stop;
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaEventCreate(&start));
  GPUCHECK(cudaEventCreate(&stop));

  GPUCHECK(cudaEventRecord(start));
  for (auto iLaunch{0}; iLaunch < nLaunches; ++iLaunch) { // Schedule all the requested kernel launches
    (*kernel)<<<blocks, threads, 0, 0>>>(args...);
  }
  GPUCHECK(cudaEventRecord(stop)); // record checkpoint

  GPUCHECK(cudaEventSynchronize(stop)); // synchronize executions
  float milliseconds{0.f};
  GPUCHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  GPUCHECK(cudaEventDestroy(start));
  GPUCHECK(cudaEventDestroy(stop));

  return milliseconds;
}

template <class chunk_type>
template <typename... T>
std::vector<float> GPUbenchmark<chunk_type>::benchmarkAsync(void (*kernel)(int, T...),
                                                            int nStreams, int nLaunches, int blocks, int threads, T&... args)
{
  std::vector<cudaEvent_t> starts(nStreams), stops(nStreams);
  std::vector<cudaStream_t> streams(nStreams);
  std::vector<float> results(nStreams);
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  for (auto iStream{0}; iStream < nStreams; ++iStream) { // one stream per chunk
    GPUCHECK(cudaStreamCreate(&(streams.at(iStream))));
    GPUCHECK(cudaEventCreate(&(starts[iStream])));
    GPUCHECK(cudaEventCreate(&(stops[iStream])));
  }

  for (auto iStream{0}; iStream < nStreams; ++iStream) {
    GPUCHECK(cudaEventRecord(starts[iStream], streams[iStream]));

    for (auto iLaunch{0}; iLaunch < 10 * nLaunches; ++iLaunch) { // 10x consecutive launches on the same stream
      (*kernel)<<<blocks, threads, 0, streams[iStream]>>>(iStream, args...);
    }
    GPUCHECK(cudaEventRecord(stops[iStream], streams[iStream]));
  }

  for (auto iStream{0}; iStream < nStreams; ++iStream) {
    GPUCHECK(cudaEventSynchronize(stops[iStream]));
    GPUCHECK(cudaEventElapsedTime(&(results.at(iStream)), starts[iStream], stops[iStream]));
    GPUCHECK(cudaEventDestroy(starts[iStream]));
    GPUCHECK(cudaEventDestroy(stops[iStream]));
    GPUCHECK(cudaStreamDestroy(streams[iStream]));
  }

  return results;
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::printDevices()
{
  int deviceCnt;
  GPUCHECK(cudaGetDeviceCount(&deviceCnt));

  for (int i = 0; i < deviceCnt; i++) {
    GPUCHECK(cudaSetDevice(i));
    printDeviceProp(i);
  }
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::globalInit()
{
  cudaDeviceProp props;
  size_t free;

  // Fetch and store features
  GPUCHECK(cudaGetDeviceProperties(&props, mOptions.deviceId));
  GPUCHECK(cudaMemGetInfo(&free, &mState.totalMemory));
  GPUCHECK(cudaSetDevice(mOptions.deviceId));

  mState.chunkReservedGB = mOptions.chunkReservedGB;
  mState.iterations = mOptions.kernelLaunches;
  mState.nMultiprocessors = props.multiProcessorCount;
  mState.nMaxThreadsPerBlock = props.maxThreadsPerMultiProcessor;
  mState.nMaxThreadsPerDimension = props.maxThreadsDim[0];
  mState.scratchSize = static_cast<long int>(mOptions.freeMemoryFractionToAllocate * free);
  std::cout << ">>> Running on: \033[1;31m" << props.name << "\e[0m" << std::endl;

  // Allocate scratch on GPU
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&mState.scratchPtr), mState.scratchSize));

  mState.computeScratchPtrs();
  GPUCHECK(cudaMemset(mState.scratchPtr, 0, mState.scratchSize))

  std::cout << "    ├ Buffer type: \e[1m" << getType<chunk_type>() << "\e[0m" << std::endl
            << "    ├ Allocated: " << std::setprecision(2) << bytesToGB(mState.scratchSize) << "/" << std::setprecision(2) << bytesToGB(mState.totalMemory)
            << "(GB) [" << std::setprecision(3) << (100.f) * (mState.scratchSize / (float)mState.totalMemory) << "%]\n"
            << "    ├ Number of scratch chunks: " << mState.getMaxChunks() << " of " << mOptions.chunkReservedGB << "GB each\n"
            << "    └ Each chunk can store up to: " << mState.getPartitionCapacity() << " elements" << std::endl
            << std::endl;
}

/// Read
template <class chunk_type>
void GPUbenchmark<chunk_type>::readInit()
{
  std::cout << ">>> Initializing read benchmarks with \e[1m" << mOptions.nTests << "\e[0m runs and \e[1m" << mOptions.kernelLaunches << "\e[0m kernel launches" << std::endl;
  mState.hostReadResultsVector.resize(mState.getMaxChunks());
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&(mState.deviceReadResultsPtr)), mState.getMaxChunks() * sizeof(chunk_type)));
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::readSequential(SplitLevel sl)
{
  switch (sl) {
    case SplitLevel::Blocks: {
      mResultWriter.get()->addBenchmarkEntry("seq_read_SB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) { // loop on the number of times we perform same measurement
        std::cout << std::setw(2) << "    ├ (" << getType<chunk_type>() << ") Seq read, sing block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        for (auto iChunk{0}; iChunk < mState.getMaxChunks(); ++iChunk) { // loop over single chunks separately
          auto result = benchmarkSync(&gpu::readChunkSBKernel<chunk_type>,
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      iChunk,
                                      mState.deviceReadResultsPtr,
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
          mResultWriter.get()->storeBenchmarkEntry(iChunk, result);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }

    case SplitLevel::Threads: {
      mResultWriter.get()->addBenchmarkEntry("seq_read_MB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) { // loop on the number of times we perform same measurement
        std::cout << std::setw(2) << "    ├ (" << getType<chunk_type>() << ") Seq read, mult block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        for (auto iChunk{0}; iChunk < mState.getMaxChunks(); ++iChunk) { // loop over single chunks separately
          auto result = benchmarkSync(&gpu::readChunkMBKernel<chunk_type>,
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      iChunk,
                                      mState.deviceReadResultsPtr,
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
          mResultWriter.get()->storeBenchmarkEntry(iChunk, result);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
  }
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::readConcurrent(SplitLevel sl, int nRegions)
{
  switch (sl) {
    case SplitLevel::Blocks: {
      mResultWriter.get()->addBenchmarkEntry("conc_read_SB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto chunks{mState.getMaxChunks()};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
        std::cout << "    ├ (" << getType<chunk_type>() << ") Conc read, sing block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        auto results = benchmarkAsync(&gpu::readChunkSBKernel<chunk_type>,
                                      mState.getMaxChunks(), // nStreams
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      mState.deviceReadResultsPtr, // kernel arguments (chunkId is passed by wrapper)
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
        for (auto iResult{0}; iResult < results.size(); ++iResult) {
          mResultWriter.get()->storeBenchmarkEntry(iResult, results[iResult]);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
    case SplitLevel::Threads: {
      mResultWriter.get()->addBenchmarkEntry("conc_read_MB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto chunks{mState.getMaxChunks()};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
        std::cout << "    ├ (" << getType<chunk_type>() << ") Conc read, mult block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        auto results = benchmarkAsync(&gpu::readChunkMBKernel<chunk_type>,
                                      mState.getMaxChunks(), // nStreams
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      mState.deviceReadResultsPtr, // kernel arguments (chunkId is passed by wrapper)
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
        for (auto iResult{0}; iResult < results.size(); ++iResult) {
          mResultWriter.get()->storeBenchmarkEntry(iResult, results[iResult]);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
  }
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::readFinalize()
{
  GPUCHECK(cudaMemcpy(mState.hostReadResultsVector.data(), mState.deviceReadResultsPtr, mState.getMaxChunks() * sizeof(chunk_type), cudaMemcpyDeviceToHost));
  GPUCHECK(cudaFree(mState.deviceReadResultsPtr));
  std::cout << "    └ done." << std::endl;
}

/// Write
template <class chunk_type>
void GPUbenchmark<chunk_type>::writeInit()
{
  std::cout << ">>> Initializing write benchmarks with \e[1m" << mOptions.nTests << "\e[0m runs and \e[1m" << mOptions.kernelLaunches << "\e[0m kernel launches" << std::endl;
  mState.hostWriteResultsVector.resize(mState.getMaxChunks());
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&(mState.deviceWriteResultsPtr)), mState.getMaxChunks() * sizeof(chunk_type)));
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::writeSequential(SplitLevel sl)
{
  switch (sl) {
    case SplitLevel::Blocks: {
      mResultWriter.get()->addBenchmarkEntry("seq_write_SB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) { // loop on the number of times we perform same measurement
        std::cout << std::setw(2) << "    ├ (" << getType<chunk_type>() << ") Seq write, sing block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        for (auto iChunk{0}; iChunk < mState.getMaxChunks(); ++iChunk) { // loop over single chunks separately
          auto result = benchmarkSync(&gpu::writeChunkSBKernel<chunk_type>,
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      iChunk,
                                      mState.deviceWriteResultsPtr,
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
          mResultWriter.get()->storeBenchmarkEntry(iChunk, result);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }

    case SplitLevel::Threads: {
      mResultWriter.get()->addBenchmarkEntry("seq_write_MB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) { // loop on the number of times we perform same measurement
        std::cout << std::setw(2) << "    ├ (" << getType<chunk_type>() << ") Seq write, mult block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        for (auto iChunk{0}; iChunk < mState.getMaxChunks(); ++iChunk) { // loop over single chunks separately
          auto result = benchmarkSync(&gpu::writeChunkMBKernel<chunk_type>,
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      iChunk,
                                      mState.deviceWriteResultsPtr,
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
          mResultWriter.get()->storeBenchmarkEntry(iChunk, result);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
  }
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::writeConcurrent(SplitLevel sl, int nRegions)
{
  switch (sl) {
    case SplitLevel::Blocks: {
      mResultWriter.get()->addBenchmarkEntry("conc_write_SB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto chunks{mState.getMaxChunks()};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
        std::cout << "    ├ (" << getType<chunk_type>() << ") Conc write, sing block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        auto results = benchmarkAsync(&gpu::writeChunkSBKernel<chunk_type>,
                                      mState.getMaxChunks(), // nStreams
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      mState.deviceWriteResultsPtr, // kernel arguments (chunkId is passed by wrapper)
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
        for (auto iResult{0}; iResult < results.size(); ++iResult) {
          mResultWriter.get()->storeBenchmarkEntry(iResult, results[iResult]);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
    case SplitLevel::Threads: {
      mResultWriter.get()->addBenchmarkEntry("conc_write_MB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto chunks{mState.getMaxChunks()};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
        std::cout << "    ├ (" << getType<chunk_type>() << ") Conc write, mult block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        auto results = benchmarkAsync(&gpu::writeChunkMBKernel<chunk_type>,
                                      mState.getMaxChunks(), // nStreams
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      mState.deviceWriteResultsPtr, // kernel arguments (chunkId is passed by wrapper)
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
        for (auto iResult{0}; iResult < results.size(); ++iResult) {
          mResultWriter.get()->storeBenchmarkEntry(iResult, results[iResult]);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
  }
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::writeFinalize()
{
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaMemcpy(mState.hostWriteResultsVector.data(), mState.deviceWriteResultsPtr, mState.getMaxChunks() * sizeof(chunk_type), cudaMemcpyDeviceToHost));
  GPUCHECK(cudaFree(mState.deviceWriteResultsPtr));
  std::cout << "    └ done." << std::endl;
}

/// Copy
template <class chunk_type>
void GPUbenchmark<chunk_type>::copyInit()
{
  std::cout << ">>> Initializing copy benchmarks with \e[1m" << mOptions.nTests << "\e[0m runs and \e[1m" << mOptions.kernelLaunches << "\e[0m kernel launches" << std::endl;
  mState.hostCopyInputsVector.resize(mState.getMaxChunks());
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaMalloc(reinterpret_cast<void**>(&(mState.deviceCopyInputsPtr)), mState.getMaxChunks() * sizeof(chunk_type)));
  GPUCHECK(cudaMemset(mState.deviceCopyInputsPtr, 1, mState.getMaxChunks() * sizeof(chunk_type)));
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::copySequential(SplitLevel sl)
{
  switch (sl) {
    case SplitLevel::Blocks: {
      mResultWriter.get()->addBenchmarkEntry("seq_copy_SB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) { // loop on the number of times we perform same measurement
        std::cout << std::setw(2) << "    ├ (" << getType<chunk_type>() << ") Seq copy, sing block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        for (auto iChunk{0}; iChunk < mState.getMaxChunks(); ++iChunk) { // loop over single chunks separately
          auto result = benchmarkSync(&gpu::copyChunkSBKernel<chunk_type>,
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      iChunk,
                                      mState.deviceCopyInputsPtr,
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
          mResultWriter.get()->storeBenchmarkEntry(iChunk, result);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }

    case SplitLevel::Threads: {
      mResultWriter.get()->addBenchmarkEntry("seq_copy_MB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) { // loop on the number of times we perform same measurement
        std::cout << std::setw(2) << "    ├ (" << getType<chunk_type>() << ") Seq copy, mult block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        for (auto iChunk{0}; iChunk < mState.getMaxChunks(); ++iChunk) { // loop over single chunks separately
          auto result = benchmarkSync(&gpu::copyChunkMBKernel<chunk_type>,
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      iChunk,
                                      mState.deviceCopyInputsPtr,
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
          mResultWriter.get()->storeBenchmarkEntry(iChunk, result);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
  }
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::copyConcurrent(SplitLevel sl, int nRegions)
{
  switch (sl) {
    case SplitLevel::Blocks: {
      mResultWriter.get()->addBenchmarkEntry("conc_copy_SB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto chunks{mState.getMaxChunks()};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
        std::cout << "    ├ (" << getType<chunk_type>() << ") Conc copy, sing block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        auto results = benchmarkAsync(&gpu::copyChunkSBKernel<chunk_type>,
                                      mState.getMaxChunks(), // nStreams
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      mState.deviceCopyInputsPtr, // kernel arguments (chunkId is passed by wrapper)
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
        for (auto iResult{0}; iResult < results.size(); ++iResult) {
          mResultWriter.get()->storeBenchmarkEntry(iResult, results[iResult]);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
    case SplitLevel::Threads: {
      mResultWriter.get()->addBenchmarkEntry("conc_copy_MB", getType<chunk_type>(), mState.getMaxChunks());
      auto nBlocks{mState.nMultiprocessors};
      auto nThreads{std::min(mState.nMaxThreadsPerDimension, mState.nMaxThreadsPerBlock)};
      auto chunks{mState.getMaxChunks()};
      auto capacity{mState.getPartitionCapacity()};

      for (auto measurement{0}; measurement < mOptions.nTests; ++measurement) {
        std::cout << "    ├ (" << getType<chunk_type>() << ") Conc copy, mult block (" << measurement + 1 << "/" << mOptions.nTests << "):";
        auto results = benchmarkAsync(&gpu::copyChunkMBKernel<chunk_type>,
                                      mState.getMaxChunks(), // nStreams
                                      mState.getNKernelLaunches(),
                                      nBlocks,
                                      nThreads,
                                      mState.deviceCopyInputsPtr, // kernel arguments (chunkId is passed by wrapper)
                                      mState.scratchPtr,
                                      capacity,
                                      mState.chunkReservedGB);
        for (auto iResult{0}; iResult < results.size(); ++iResult) {
          auto region = getCorrespondingRegionId(iResult, nBlocks, nRegions);
          mResultWriter.get()->storeBenchmarkEntry(iResult, results[iResult]);
        }
        mResultWriter.get()->snapshotBenchmark();
        std::cout << "\033[1;32m complete\033[0m" << std::endl;
      }
      break;
    }
  }
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::copyFinalize()
{
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaMemcpy(mState.hostCopyInputsVector.data(), mState.deviceCopyInputsPtr, mState.getMaxChunks() * sizeof(chunk_type), cudaMemcpyDeviceToHost));
  GPUCHECK(cudaFree(mState.deviceCopyInputsPtr));
  std::cout << "    └ done." << std::endl;
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::globalFinalize()
{
  GPUCHECK(cudaSetDevice(mOptions.deviceId));
  GPUCHECK(cudaFree(mState.scratchPtr));
}

template <class chunk_type>
void GPUbenchmark<chunk_type>::run()
{
  globalInit();

  for (auto& sl : mOptions.pools) {
    for (auto& test : mOptions.tests) {
      switch (test) {
        case Test::Read: {
          readInit();

          if (std::find(mOptions.modes.begin(), mOptions.modes.end(), Mode::Sequential) != mOptions.modes.end()) {
            readSequential(sl);
          }
          if (std::find(mOptions.modes.begin(), mOptions.modes.end(), Mode::Concurrent) != mOptions.modes.end()) {
            readConcurrent(sl);
          }

          readFinalize();

          break;
        }
        case Test::Write: {
          writeInit();
          if (std::find(mOptions.modes.begin(), mOptions.modes.end(), Mode::Sequential) != mOptions.modes.end()) {
            writeSequential(sl);
          }
          if (std::find(mOptions.modes.begin(), mOptions.modes.end(), Mode::Concurrent) != mOptions.modes.end()) {
            writeConcurrent(sl);
          }

          writeFinalize();

          break;
        }
        case Test::Copy: {
          copyInit();
          if (std::find(mOptions.modes.begin(), mOptions.modes.end(), Mode::Sequential) != mOptions.modes.end()) {
            copySequential(sl);
          }
          if (std::find(mOptions.modes.begin(), mOptions.modes.end(), Mode::Concurrent) != mOptions.modes.end()) {
            copyConcurrent(sl);
          }

          copyFinalize();

          break;
        }
      }
    }
  }

  globalFinalize();
}

template class GPUbenchmark<char>;
template class GPUbenchmark<size_t>;
template class GPUbenchmark<int>;
// template class GPUbenchmark<uint4>;

} // namespace benchmark
} // namespace o2
