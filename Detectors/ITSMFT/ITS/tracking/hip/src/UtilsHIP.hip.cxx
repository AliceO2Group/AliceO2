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
/// \file utilsHIP.hip.cxx
/// \brief
///

#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime_api.h>
#include "ITStrackingHIP/ContextHIP.h"
#include "ITStrackingHIP/UtilsHIP.h"

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

void utils::host_hip::checkHIPError(const hipError_t error, const char* file, const int line)
{
  if (error != hipSuccess) {
    std::ostringstream errorString{};
    errorString << file << ":" << line << " HIP API returned error [" << hipGetErrorString(error) << "] (code "
                << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

dim3 utils::host_hip::getBlockSize(const int colsNum)
{
  return getBlockSize(colsNum, 1);
}

dim3 utils::host_hip::getBlockSize(const int colsNum, const int rowsNum)
{
  const DeviceProperties& deviceProperties = ContextHIP::getInstance().getDeviceProperties();
  return getBlockSize(colsNum, rowsNum, deviceProperties.streamProcessors / deviceProperties.maxBlocksPerSM);
}

dim3 utils::host_hip::getBlockSize(const int colsNum, const int rowsNum, const int maxThreadsPerBlock)
{
  const DeviceProperties& deviceProperties = ContextHIP::getInstance().getDeviceProperties();
  int xThreads = std::max(std::min(colsNum, static_cast<int>(deviceProperties.maxThreadsDim.x)), 1);
  int yThreads = std::max(std::min(rowsNum, static_cast<int>(deviceProperties.maxThreadsDim.y)), 1);
  const int totalThreads = roundUp(std::min(xThreads * yThreads, maxThreadsPerBlock),
                                   static_cast<int>(deviceProperties.warpSize));
  if (xThreads > yThreads) {
    xThreads = findNearestDivisor(xThreads, totalThreads);
    yThreads = totalThreads / xThreads;

  } else {
    yThreads = findNearestDivisor(yThreads, totalThreads);
    xThreads = totalThreads / yThreads;
  }

  return dim3{static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads)};
}

dim3 utils::host_hip::getBlocksGrid(const dim3& threadsPerBlock, const int rowsNum)
{
  return getBlocksGrid(threadsPerBlock, rowsNum, 1);
}

dim3 utils::host_hip::getBlocksGrid(const dim3& threadsPerBlock, const int rowsNum, const int colsNum)
{
  return dim3{1 + (rowsNum - 1) / threadsPerBlock.x, 1 + (colsNum - 1) / threadsPerBlock.y};
}

void utils::host_hip::gpuMalloc(void** p, const int size)
{
  checkHIPError(hipMalloc(p, size), __FILE__, __LINE__);
}

void utils::host_hip::gpuFree(void* p)
{
  checkHIPError(hipFree(p), __FILE__, __LINE__);
}

void utils::host_hip::gpuMemset(void* p, int value, int size)
{
  checkHIPError(hipMemset(p, value, size), __FILE__, __LINE__);
}

void utils::host_hip::gpuMemcpyHostToDevice(void* dst, const void* src, int size)
{
  checkHIPError(hipMemcpy(dst, src, size, hipMemcpyHostToDevice), __FILE__, __LINE__);
}

void utils::host_hip::gpuMemcpyHostToDeviceAsync(void* dst, const void* src, int size, hipStream_t& stream)
{
  checkHIPError(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream), __FILE__, __LINE__);
}

void utils::host_hip::gpuMemcpyDeviceToHost(void* dst, const void* src, int size)
{
  checkHIPError(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost), __FILE__, __LINE__);
}

// void utils::host_hip::gpuStartProfiler()
// {
//   checkHIPError(hipProfilerStart(), __FILE__, __LINE__);
// }

// void utils::host_hip::gpuStopProfiler()
// {
//   checkHIPError(hipProfilerStop(), __FILE__, __LINE__);
// }

GPUd() int utils::device_hip::getLaneIndex()
{
  uint32_t laneIndex;
  asm volatile("mov.u32 %0, %%laneid;"
               : "=r"(laneIndex));
  return static_cast<int>(laneIndex);
}

// GPUd() int utils::Device::shareToWarp(const int value, const int laneIndex)
// {
//   cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
//   return threadGroup.shfl(value, laneIndex);
// }

// GPUd() int utils::Device::gpuAtomicAdd(int* p, const int incrementSize)
// {
//   return atomicAdd(p, incrementSize);
// }

} // namespace gpu
} // namespace its
} // namespace o2
