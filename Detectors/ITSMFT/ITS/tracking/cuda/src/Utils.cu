// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Utils.cu
/// \brief
///

#include "ITStrackingCUDA/Utils.h"

#include <sstream>
#include <stdexcept>

#ifndef GPUCA_GPUCODE_GENRTC
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#endif

#include "ITStrackingCUDA/Context.h"

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
namespace gpu
{

void utils::host::checkCUDAError(const cudaError_t error, const char* file, const int line)
{
  if (error != cudaSuccess) {

    std::ostringstream errorString{};

    errorString << file << ":" << line << " CUDA API returned error [" << cudaGetErrorString(error) << "] (code "
                << error << ")" << std::endl;

    throw std::runtime_error{errorString.str()};
  }
}

dim3 utils::host::getBlockSize(const int colsNum)
{
  return getBlockSize(colsNum, 1);
}

dim3 utils::host::getBlockSize(const int colsNum, const int rowsNum)
{
  const DeviceProperties& deviceProperties = Context::getInstance().getDeviceProperties();
  return getBlockSize(colsNum, rowsNum, deviceProperties.cudaCores / deviceProperties.maxBlocksPerSM);
}

dim3 utils::host::getBlockSize(const int colsNum, const int rowsNum, const int maxThreadsPerBlock)
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

dim3 utils::host::getBlocksGrid(const dim3& threadsPerBlock, const int rowsNum)
{

  return getBlocksGrid(threadsPerBlock, rowsNum, 1);
}

dim3 utils::host::getBlocksGrid(const dim3& threadsPerBlock, const int rowsNum, const int colsNum)
{

  return dim3{1 + (rowsNum - 1) / threadsPerBlock.x, 1 + (colsNum - 1) / threadsPerBlock.y};
}

void utils::host::gpuMalloc(void** p, const int size)
{
  checkCUDAError(cudaMalloc(p, size), __FILE__, __LINE__);
}

void utils::host::gpuFree(void* p)
{
  checkCUDAError(cudaFree(p), __FILE__, __LINE__);
}

void utils::host::gpuMemset(void* p, int value, int size)
{
  checkCUDAError(cudaMemset(p, value, size), __FILE__, __LINE__);
}

void utils::host::gpuMemcpyHostToDevice(void* dst, const void* src, int size)
{
  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void utils::host::gpuMemcpyHostToDeviceAsync(void* dst, const void* src, int size, Stream& stream)
{
  checkCUDAError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream.get()), __FILE__, __LINE__);
}

void utils::host::gpuMemcpyDeviceToHost(void* dst, const void* src, int size)
{
  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

// void utils::host::gpuStartProfiler()
// {
//   checkCUDAError(cudaProfilerStart(), __FILE__, __LINE__);
// }

// void utils::host::gpuStopProfiler()
// {
//   checkCUDAError(cudaProfilerStop(), __FILE__, __LINE__);
// }

GPUd() int utils::device::getLaneIndex()
{
  uint32_t laneIndex;
  asm volatile("mov.u32 %0, %%laneid;"
               : "=r"(laneIndex));
  return static_cast<int>(laneIndex);
}

GPUd() int utils::device::shareToWarp(const int value, const int laneIndex)
{
  cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
  return threadGroup.shfl(value, laneIndex);
}

GPUd() int utils::device::gpuAtomicAdd(int* p, const int incrementSize)
{
  return atomicAdd(p, incrementSize);
}

} // namespace gpu
} // namespace its
} // namespace o2
