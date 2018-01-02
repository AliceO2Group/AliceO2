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
/// \file CAGPUtils.cu
/// \brief
///

#include "ITSReconstruction/CA/gpu/Utils.h"

#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>

#include "ITSReconstruction/CA/gpu/Context.h"

namespace {
void checkCUDAError(const cudaError_t error, const char *file, const int line)
{
  if (error != cudaSuccess) {

    std::ostringstream errorString { };

    errorString << file << ":" << line << " CUDA API returned error [" << cudaGetErrorString(error) << "] (code "
        << error << ")" << std::endl;

    throw std::runtime_error { errorString.str() };
  }
}

int roundUp(const int numToRound, const int multiple)
{
  if (multiple == 0) {

    return numToRound;
  }

  int remainder { numToRound % multiple };

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

}

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

dim3 Utils::Host::getBlockSize(const int colsNum)
{
  return getBlockSize(colsNum, 1);
}

dim3 Utils::Host::getBlockSize(const int colsNum, const int rowsNum)
{
  const DeviceProperties& deviceProperties = Context::getInstance().getDeviceProperties();
  return getBlockSize(colsNum, rowsNum, deviceProperties.cudaCores / deviceProperties.maxBlocksPerSM);
}

dim3 Utils::Host::getBlockSize(const int colsNum, const int rowsNum, const int maxThreadsPerBlock)
{
  const DeviceProperties& deviceProperties = Context::getInstance().getDeviceProperties();
  int xThreads = min(colsNum, deviceProperties.maxThreadsDim.x);
  int yThreads = min(rowsNum, deviceProperties.maxThreadsDim.y);
  const int totalThreads = roundUp(min(xThreads * yThreads, maxThreadsPerBlock),
      deviceProperties.warpSize);

  if (xThreads > yThreads) {

    xThreads = findNearestDivisor(xThreads, totalThreads);
    yThreads = totalThreads / xThreads;

  } else {

    yThreads = findNearestDivisor(yThreads, totalThreads);
    xThreads = totalThreads / yThreads;
  }

  return dim3 { static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads) };
}

dim3 Utils::Host::getBlocksGrid(const dim3 &threadsPerBlock, const int rowsNum)
{

  return getBlocksGrid(threadsPerBlock, rowsNum, 1);
}

dim3 Utils::Host::getBlocksGrid(const dim3 &threadsPerBlock, const int rowsNum, const int colsNum)
{

  return dim3 { 1 + (rowsNum - 1) / threadsPerBlock.x, 1 + (colsNum - 1) / threadsPerBlock.y };
}

void Utils::Host::gpuMalloc(void **p, const int size)
{
  checkCUDAError(cudaMalloc(p, size), __FILE__, __LINE__);
}

void Utils::Host::gpuFree(void *p)
{
  checkCUDAError(cudaFree(p), __FILE__, __LINE__);
}

void Utils::Host::gpuMemset(void *p, int value, int size)
{
  checkCUDAError(cudaMemset(p, value, size), __FILE__, __LINE__);
}

void Utils::Host::gpuMemcpyHostToDevice(void *dst, const void *src, int size)
{
  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void Utils::Host::gpuMemcpyHostToDeviceAsync(void *dst, const void *src, int size, Stream &stream)
{
  checkCUDAError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream.get()), __FILE__, __LINE__);
}

void Utils::Host::gpuMemcpyDeviceToHost(void *dst, const void *src, int size)
{
  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

void Utils::Host::gpuStartProfiler()
{
  checkCUDAError(cudaProfilerStart(), __FILE__, __LINE__);
}

void Utils::Host::gpuStopProfiler()
{
  checkCUDAError(cudaProfilerStop(), __FILE__, __LINE__);
}

GPU_DEVICE int Utils::Device::getLaneIndex()
{
  uint32_t laneIndex;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneIndex));
  return static_cast<int>(laneIndex);
}

GPU_DEVICE int Utils::Device::shareToWarp(const int value, const int laneIndex)
{
<<<<<<< HEAD
  cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
  return threadGroup.shfl(value, laneIndex);
=======
  return __shfl(value, laneIndex);
>>>>>>> [WIP] Add tracking-itsu repository content
}

GPU_DEVICE int Utils::Device::gpuAtomicAdd(int *p, const int incrementSize)
{
  return atomicAdd(p, incrementSize);
}

}
}
}
}
