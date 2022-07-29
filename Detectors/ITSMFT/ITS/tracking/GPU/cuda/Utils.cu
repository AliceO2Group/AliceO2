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

#include <sstream>
#include <stdexcept>

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

void utils::host::checkGPUError(const cudaError_t error, const char* file, const int line)
{
  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << file << ":" << line << std::endl
                << GPU_ARCH << " API returned error [" << cudaGetErrorString(error) << "] (code "
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
  return getBlockSize(colsNum, rowsNum, deviceProperties.gpuCores / deviceProperties.maxBlocksPerSM);
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
  checkGPUError(cudaMalloc(p, size), __FILE__, __LINE__);
}

void utils::host::gpuFree(void* p)
{
  checkGPUError(cudaFree(p), __FILE__, __LINE__);
}

void utils::host::gpuMemset(void* p, int value, int size)
{
  checkGPUError(cudaMemset(p, value, size), __FILE__, __LINE__);
}

void utils::host::gpuMemcpyHostToDevice(void* dst, const void* src, int size)
{
  checkGPUError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void utils::host::gpuMemcpyDeviceToHost(void* dst, const void* src, int size)
{
  checkGPUError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

void utils::host::gpuMemcpyToSymbol(const void* symbol, const void* src, int size)
{
  checkGPUError(cudaMemcpyToSymbol(symbol, src, size, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void utils::host::gpuMemcpyFromSymbol(void* dst, const void* symbol, int size)
{
  checkGPUError(cudaMemcpyFromSymbol(dst, symbol, size, 0, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

GPUd() int utils::device::getLaneIndex()
{
  uint32_t laneIndex;
  asm volatile("mov.u32 %0, %%laneid;"
               : "=r"(laneIndex));
  return static_cast<int>(laneIndex);
}
} // namespace gpu
} // namespace its
} // namespace o2
