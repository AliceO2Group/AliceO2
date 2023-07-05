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

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/Context.h"
#include "ITStracking/Constants.h"

#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <iomanip>
#include <numeric>
#include <iostream>

namespace
{
int roundUp(const int numToRound, const int multiple)
{
  if (multiple == 0) {
    return numToRound;
  }

  int remainder{numToRound % multiple};
  if (remainder == 0) {
    return numToRound;
  }
  return numToRound + multiple - remainder;
}

int findNearestDivisor(const int numToRound, const int divisor)
{

  if (numToRound > divisor) {
    return divisor;
  }

  int result = numToRound;
  while (divisor % result != 0) {
    ++result;
  }
  return result;
}

} // namespace

namespace o2
{
namespace its
{
using constants::GB;
namespace gpu
{
GPUh() void gpuThrowOnError()
{
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << GPU_ARCH << " API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

double bytesToconfig(size_t s) { return (double)s / (1024.0); }
double bytesToGB(size_t s) { return (double)s / GB; }

void utils::checkGPUError(const cudaError_t error, const char* file, const int line)
{
  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << file << ":" << line << std::endl
                << GPU_ARCH << " API returned error [" << cudaGetErrorString(error) << "] (code "
                << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

void utils::getDeviceProp(int deviceId, bool print)
{
  const int w1 = 34;
  std::cout << std::left;
  std::cout << std::setw(w1)
            << "--------------------------------------------------------------------------------"
            << std::endl;
  std::cout << std::setw(w1) << "device#" << deviceId << std::endl;

  cudaDeviceProp props;
  checkGPUError(cudaGetDeviceProperties(&props, deviceId));
  if (print) {
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
    checkGPUError(cudaGetDeviceCount(&deviceCnt));
    std::cout << std::setw(w1) << "peers: ";
    for (int i = 0; i < deviceCnt; i++) {
      int isPeer;
      checkGPUError(cudaDeviceCanAccessPeer(&isPeer, i, deviceId));
      if (isPeer) {
        std::cout << "device#" << i << " ";
      }
    }
    std::cout << std::endl;
    std::cout << std::setw(w1) << "non-peers: ";
    for (int i = 0; i < deviceCnt; i++) {
      int isPeer;
      checkGPUError(cudaDeviceCanAccessPeer(&isPeer, i, deviceId));
      if (!isPeer) {
        std::cout << "device#" << i << " ";
      }
    }
    std::cout << std::endl;

    size_t free, total;
    checkGPUError(cudaMemGetInfo(&free, &total));

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(w1) << "memInfo.total: " << bytesToGB(total) << " GB" << std::endl;
    std::cout << std::setw(w1) << "memInfo.free:  " << bytesToGB(free) << " GB (" << std::setprecision(0)
              << (float)free / total * 100.0 << "%)" << std::endl;
  }
}

dim3 utils::getBlockSize(const int colsNum)
{
  return getBlockSize(colsNum, 1);
}

dim3 utils::getBlockSize(const int colsNum, const int rowsNum)
{
  const DeviceProperties& deviceProperties = Context::getInstance().getDeviceProperties();
  return getBlockSize(colsNum, rowsNum, deviceProperties.gpuCores / deviceProperties.maxBlocksPerSM);
}

dim3 utils::getBlockSize(const int colsNum, const int rowsNum, const int maxThreadsPerBlock)
{
  const DeviceProperties& deviceProperties = Context::getInstance().getDeviceProperties();
  int xThreads = max(min(colsNum, deviceProperties.maxThreadsDim.x), 1);
  int yThreads = max(min(rowsNum, deviceProperties.maxThreadsDim.y), 1);
  const int totalThreads = roundUp(min(xThreads * yThreads, maxThreadsPerBlock),
                                   deviceProperties.warpSize);

  if (xThreads > yThreads) {

    xThreads = findNearestDivisor(xThreads, totalThreads);
    yThreads = totalThreads / xThreads;

  } else {

    yThreads = findNearestDivisor(yThreads, totalThreads);
    xThreads = totalThreads / yThreads;
  }

  return dim3{static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads)};
}

dim3 utils::getBlocksGrid(const dim3& threadsPerBlock, const int rowsNum)
{
  return getBlocksGrid(threadsPerBlock, rowsNum, 1);
}

dim3 utils::getBlocksGrid(const dim3& threadsPerBlock, const int rowsNum, const int colsNum)
{

  return dim3{1 + (rowsNum - 1) / threadsPerBlock.x, 1 + (colsNum - 1) / threadsPerBlock.y};
}

void utils::gpuMalloc(void** p, const int size)
{
  checkGPUError(cudaMalloc(p, size), __FILE__, __LINE__);
}

void utils::gpuFree(void* p)
{
  checkGPUError(cudaFree(p), __FILE__, __LINE__);
}

void utils::gpuMemset(void* p, int value, int size)
{
  checkGPUError(cudaMemset(p, value, size), __FILE__, __LINE__);
}

void utils::gpuMemcpyHostToDevice(void* dst, const void* src, int size)
{
  checkGPUError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void utils::gpuMemcpyDeviceToHost(void* dst, const void* src, int size)
{
  checkGPUError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

void utils::gpuMemcpyToSymbol(const void* symbol, const void* src, int size)
{
  checkGPUError(cudaMemcpyToSymbol(symbol, src, size, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void utils::gpuMemcpyFromSymbol(void* dst, const void* symbol, int size)
{
  checkGPUError(cudaMemcpyFromSymbol(dst, symbol, size, 0, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

GPUd() int utils::getLaneIndex()
{
  uint32_t laneIndex;
  asm volatile("mov.u32 %0, %%laneid;"
               : "=r"(laneIndex));
  return static_cast<int>(laneIndex);
}
} // namespace gpu
} // namespace its
} // namespace o2
