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

#include "ITStrackingGPU/Context.h"
#include "ITStrackingGPU/Utils.h"

#include <sstream>
#include <stdexcept>
#include <iostream>

namespace o2
{
namespace its
{
namespace gpu
{

using utils::checkGPUError;

Context::Context(bool dumpDevices)
{
  checkGPUError(cudaGetDeviceCount(&mDevicesNum), __FILE__, __LINE__);

  if (mDevicesNum == 0) {
    throw std::runtime_error{"There are no available GPU device(s)\n"};
  }

  mDeviceProperties.resize(mDevicesNum, DeviceProperties{});

  int currentDeviceIndex;
  checkGPUError(cudaGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  for (int iDevice{0}; iDevice < mDevicesNum; ++iDevice) {

    cudaDeviceProp deviceProperties;

    checkGPUError(cudaSetDevice(iDevice), __FILE__, __LINE__);
    checkGPUError(cudaGetDeviceProperties(&deviceProperties, iDevice), __FILE__, __LINE__);

    int major = deviceProperties.major;
    int minor = deviceProperties.minor;

    mDeviceProperties[iDevice].name = deviceProperties.name;
    mDeviceProperties[iDevice].gpuProcessors = deviceProperties.multiProcessorCount;
    mDeviceProperties[iDevice].gpuCores = getGPUCores(major, minor) * deviceProperties.multiProcessorCount;
    mDeviceProperties[iDevice].globalMemorySize = deviceProperties.totalGlobalMem;
    mDeviceProperties[iDevice].constantMemorySize = deviceProperties.totalConstMem;
    mDeviceProperties[iDevice].sharedMemorySize = deviceProperties.sharedMemPerBlock;
    mDeviceProperties[iDevice].maxClockRate = deviceProperties.memoryClockRate;
    mDeviceProperties[iDevice].busWidth = deviceProperties.memoryBusWidth;
    mDeviceProperties[iDevice].l2CacheSize = deviceProperties.l2CacheSize;
    mDeviceProperties[iDevice].registersPerBlock = deviceProperties.regsPerBlock;
    mDeviceProperties[iDevice].warpSize = deviceProperties.warpSize;
    mDeviceProperties[iDevice].maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    mDeviceProperties[iDevice].maxBlocksPerSM = getGPUMaxThreadsPerComputingUnit();
    mDeviceProperties[iDevice].maxThreadsDim = dim3{static_cast<unsigned int>(deviceProperties.maxThreadsDim[0]),
                                                    static_cast<unsigned int>(deviceProperties.maxThreadsDim[1]),
                                                    static_cast<unsigned int>(deviceProperties.maxThreadsDim[2])};
    mDeviceProperties[iDevice].maxGridDim = dim3{static_cast<unsigned int>(deviceProperties.maxGridSize[0]),
                                                 static_cast<unsigned int>(deviceProperties.maxGridSize[1]),
                                                 static_cast<unsigned int>(deviceProperties.maxGridSize[2])};
    if (dumpDevices) {
      std::cout << "################ " << GPU_ARCH << " DEVICE " << iDevice << " ################" << std::endl;
      std::cout << "Name " << mDeviceProperties[iDevice].name << std::endl;
      std::cout << "minor " << minor << " major " << major << std::endl;
      std::cout << "gpuProcessors " << mDeviceProperties[iDevice].gpuProcessors << std::endl;
      std::cout << "gpuCores " << mDeviceProperties[iDevice].gpuCores << std::endl;
      std::cout << "globalMemorySize " << mDeviceProperties[iDevice].globalMemorySize << std::endl;
      std::cout << "constantMemorySize " << mDeviceProperties[iDevice].constantMemorySize << std::endl;
      std::cout << "sharedMemorySize " << mDeviceProperties[iDevice].sharedMemorySize << std::endl;
      std::cout << "maxClockRate " << mDeviceProperties[iDevice].maxClockRate << std::endl;
      std::cout << "busWidth " << mDeviceProperties[iDevice].busWidth << std::endl;
      std::cout << "l2CacheSize " << mDeviceProperties[iDevice].l2CacheSize << std::endl;
      std::cout << "registersPerBlock " << mDeviceProperties[iDevice].registersPerBlock << std::endl;
      std::cout << "warpSize " << mDeviceProperties[iDevice].warpSize << std::endl;
      std::cout << "maxThreadsPerBlock " << mDeviceProperties[iDevice].maxThreadsPerBlock << std::endl;
      std::cout << "maxBlocksPerSM " << mDeviceProperties[iDevice].maxBlocksPerSM << std::endl;
      std::cout << "maxThreadsDim " << mDeviceProperties[iDevice].maxThreadsDim.x << ", "
                << mDeviceProperties[iDevice].maxThreadsDim.y << ", "
                << mDeviceProperties[iDevice].maxThreadsDim.z << std::endl;
      std::cout << "maxGridDim " << mDeviceProperties[iDevice].maxGridDim.x << ", "
                << mDeviceProperties[iDevice].maxGridDim.y << ", "
                << mDeviceProperties[iDevice].maxGridDim.z << std::endl;
      std::cout << std::endl;
    }
  }

  checkGPUError(cudaSetDevice(currentDeviceIndex), __FILE__, __LINE__);
}

Context& Context::getInstance()
{
  static Context gpuContext;
  return gpuContext;
}

const DeviceProperties& Context::getDeviceProperties()
{
  int currentDeviceIndex;
  checkGPUError(cudaGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  return getDeviceProperties(currentDeviceIndex);
}

const DeviceProperties& Context::getDeviceProperties(const int deviceIndex)
{
  return mDeviceProperties[deviceIndex];
}

} // namespace gpu
} // namespace its
} // namespace o2
