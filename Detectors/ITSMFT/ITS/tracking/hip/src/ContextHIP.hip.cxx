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
/// \file Context.cu
/// \brief
///

#include "ITStrackingHIP/ContextHIP.h"
#include "ITStrackingHIP/UtilsHIP.h"

#include <sstream>
#include <stdexcept>

#include <iostream>

namespace
{

inline int getHIPCores(const int major, const int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
    {
      {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
      {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
      {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
      {0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
      {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
      {0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
      {0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
      {0x60, 64},  // Pascal Generation (SM 6.0) GP100 class
      {0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
      {0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline int getMaxThreadsPerSM(const int major, const int minor)
{
  return 8;
}

} // namespace

namespace o2
{
namespace its
{
namespace GPU
{

using Utils::Host::checkHIPError;

Context::Context(bool dumpDevices)
{
  checkHIPError(hipGetDeviceCount(&mDevicesNum), __FILE__, __LINE__);

  if (mDevicesNum == 0) {

    throw std::runtime_error{"There are no available device(s) that support CUDA\n"};
  }

  mDeviceProperties.resize(mDevicesNum, DeviceProperties{});

  int currentDeviceIndex;
  checkHIPError(hipGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  for (int iDevice{0}; iDevice < mDevicesNum; ++iDevice) {

    hipDeviceProp_t deviceProperties;

    checkHIPError(hipSetDevice(iDevice), __FILE__, __LINE__);
    checkHIPError(hipGetDeviceProperties(&deviceProperties, iDevice), __FILE__, __LINE__);

    // int major = deviceProperties.major; // Codacy warning
    // int minor = deviceProperties.minor; // Codacy warning

    mDeviceProperties[iDevice].name = deviceProperties.name;
    mDeviceProperties[iDevice].gpuProcessors = deviceProperties.multiProcessorCount;
    // mDeviceProperties[iDevice].hipCores = getHIPCores(major, minor) * deviceProperties.multiProcessorCount; // >>>> alarm
    mDeviceProperties[iDevice].globalMemorySize = deviceProperties.totalGlobalMem;
    mDeviceProperties[iDevice].constantMemorySize = deviceProperties.totalConstMem;
    mDeviceProperties[iDevice].sharedMemorySize = deviceProperties.sharedMemPerBlock;
    mDeviceProperties[iDevice].maxClockRate = deviceProperties.memoryClockRate;
    mDeviceProperties[iDevice].busWidth = deviceProperties.memoryBusWidth;
    mDeviceProperties[iDevice].l2CacheSize = deviceProperties.l2CacheSize;
    mDeviceProperties[iDevice].registersPerBlock = deviceProperties.regsPerBlock;
    mDeviceProperties[iDevice].warpSize = deviceProperties.warpSize;
    mDeviceProperties[iDevice].maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    // mDeviceProperties[iDevice].maxBlocksPerSM = getMaxThreadsPerSM(major, minor);
    mDeviceProperties[iDevice].maxThreadsDim = dim3{static_cast<unsigned int>(deviceProperties.maxThreadsDim[0]),
                                                    static_cast<unsigned int>(deviceProperties.maxThreadsDim[1]),
                                                    static_cast<unsigned int>(deviceProperties.maxThreadsDim[2])};
    mDeviceProperties[iDevice].maxGridDim = dim3{static_cast<unsigned int>(deviceProperties.maxGridSize[0]),
                                                 static_cast<unsigned int>(deviceProperties.maxGridSize[1]),
                                                 static_cast<unsigned int>(deviceProperties.maxGridSize[2])};
    if (dumpDevices) {
      std::cout << "################ HIP DEVICE " << iDevice << " ################" << std::endl;
      std::cout << "Name " << mDeviceProperties[iDevice].name << std::endl;
      std::cout << "gpuProcessors " << mDeviceProperties[iDevice].gpuProcessors << std::endl; // >>>>> ALLARME
      // std::cout << "hipCores " << mDeviceProperties[iDevice].hipCores << std::endl;
      std::cout << "globalMemorySize " << mDeviceProperties[iDevice].globalMemorySize << std::endl;
      std::cout << "constantMemorySize " << mDeviceProperties[iDevice].constantMemorySize << std::endl;
      std::cout << "sharedMemorySize " << mDeviceProperties[iDevice].sharedMemorySize << std::endl;
      std::cout << "maxClockRate " << mDeviceProperties[iDevice].maxClockRate << std::endl;
      std::cout << "busWidth " << mDeviceProperties[iDevice].busWidth << std::endl;
      std::cout << "l2CacheSize " << mDeviceProperties[iDevice].l2CacheSize << std::endl;
      std::cout << "registersPerBlock " << mDeviceProperties[iDevice].registersPerBlock << std::endl;
      std::cout << "warpSize " << mDeviceProperties[iDevice].warpSize << std::endl;
      std::cout << "maxThreadsPerBlock " << mDeviceProperties[iDevice].maxThreadsPerBlock << std::endl;
      // std::cout << "maxBlocksPerSM " << mDeviceProperties[iDevice].maxBlocksPerSM << std::endl;
      std::cout << "maxThreadsDim " << mDeviceProperties[iDevice].maxThreadsDim.x << ", " << mDeviceProperties[iDevice].maxThreadsDim.y << ", " << mDeviceProperties[iDevice].maxThreadsDim.z << std::endl;
      std::cout << "maxGridDim " << mDeviceProperties[iDevice].maxGridDim.x << ", " << mDeviceProperties[iDevice].maxGridDim.y << ", " << mDeviceProperties[iDevice].maxGridDim.z << std::endl;
      std::cout << std::endl;
    }
  }

  checkHIPError(hipSetDevice(currentDeviceIndex), __FILE__, __LINE__);
}

Context& Context::getInstance()
{
  static Context gpuContext;
  return gpuContext;
}

const DeviceProperties& Context::getDeviceProperties()
{
  int currentDeviceIndex;
  checkHIPError(hipGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  return getDeviceProperties(currentDeviceIndex);
}

const DeviceProperties& Context::getDeviceProperties(const int deviceIndex)
{
  return mDeviceProperties[deviceIndex];
}

} // namespace GPU
} // namespace its
} // namespace o2
