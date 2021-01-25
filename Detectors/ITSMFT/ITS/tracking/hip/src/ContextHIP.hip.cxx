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
/// \file ContextHIP.hip.cxx
/// \brief
///

#include "ITStrackingHIP/ContextHIP.h"
#include "ITStrackingHIP/UtilsHIP.h"

#include <sstream>
#include <stdexcept>

#include <iostream>

namespace
{

inline int getStreamProcessors(const int major, const int minor)
{
  // Hardcoded result for AMD RADEON WX 9100, to be decided if and how determine this paramter
  return 4096;
}

inline int getMaxThreadsPerComputingUnit(const int major, const int minor)
{
  return 8;
}

} // namespace

namespace o2
{
namespace its
{
namespace gpu
{

using utils::host_hip::checkHIPError;

ContextHIP::ContextHIP(bool dumpDevices)
{
  checkHIPError(hipGetDeviceCount(&mDevicesNum), __FILE__, __LINE__);
  if (mDevicesNum == 0) {
    throw std::runtime_error{"There are no available device(s) that support HIP\n"};
  }

  mDeviceProperties.resize(mDevicesNum, DeviceProperties{});

  int currentDeviceIndex;
  checkHIPError(hipGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  for (int iDevice{0}; iDevice < mDevicesNum; ++iDevice) {

    hipDeviceProp_t deviceProperties;

    checkHIPError(hipSetDevice(iDevice), __FILE__, __LINE__);
    checkHIPError(hipGetDeviceProperties(&deviceProperties, iDevice), __FILE__, __LINE__);

    int major = deviceProperties.major; // Codacy warning
    int minor = deviceProperties.minor; // Codacy warning

    mDeviceProperties[iDevice].name = deviceProperties.name;
    mDeviceProperties[iDevice].gpuProcessors = deviceProperties.multiProcessorCount;
    mDeviceProperties[iDevice].streamProcessors = getStreamProcessors(major, minor) * deviceProperties.multiProcessorCount; // >>>> alarm
    mDeviceProperties[iDevice].globalMemorySize = deviceProperties.totalGlobalMem;
    mDeviceProperties[iDevice].constantMemorySize = deviceProperties.totalConstMem;
    mDeviceProperties[iDevice].sharedMemorySize = deviceProperties.sharedMemPerBlock;
    mDeviceProperties[iDevice].maxClockRate = deviceProperties.memoryClockRate;
    mDeviceProperties[iDevice].busWidth = deviceProperties.memoryBusWidth;
    mDeviceProperties[iDevice].l2CacheSize = deviceProperties.l2CacheSize;
    mDeviceProperties[iDevice].registersPerBlock = deviceProperties.regsPerBlock;
    mDeviceProperties[iDevice].warpSize = deviceProperties.warpSize;
    mDeviceProperties[iDevice].maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    mDeviceProperties[iDevice].maxBlocksPerSM = getMaxThreadsPerComputingUnit(major, minor);
    mDeviceProperties[iDevice].maxThreadsDim = dim3{static_cast<unsigned int>(deviceProperties.maxThreadsDim[0]),
                                                    static_cast<unsigned int>(deviceProperties.maxThreadsDim[1]),
                                                    static_cast<unsigned int>(deviceProperties.maxThreadsDim[2])};
    mDeviceProperties[iDevice].maxGridDim = dim3{static_cast<unsigned int>(deviceProperties.maxGridSize[0]),
                                                 static_cast<unsigned int>(deviceProperties.maxGridSize[1]),
                                                 static_cast<unsigned int>(deviceProperties.maxGridSize[2])};
    if (dumpDevices) {
      std::cout << "################ HIP DEVICE " << iDevice << " ################" << std::endl;
      std::cout << "Name " << mDeviceProperties[iDevice].name << std::endl;
      std::cout << "gpuProcessors " << mDeviceProperties[iDevice].gpuProcessors << std::endl;
      std::cout << "minor " << minor << " major " << major << std::endl;
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
      std::cout << "maxThreadsDim " << mDeviceProperties[iDevice].maxThreadsDim.x << ", " << mDeviceProperties[iDevice].maxThreadsDim.y << ", " << mDeviceProperties[iDevice].maxThreadsDim.z << std::endl;
      std::cout << "maxGridDim " << mDeviceProperties[iDevice].maxGridDim.x << ", " << mDeviceProperties[iDevice].maxGridDim.y << ", " << mDeviceProperties[iDevice].maxGridDim.z << std::endl;
      std::cout << std::endl;
    }
  }

  checkHIPError(hipSetDevice(currentDeviceIndex), __FILE__, __LINE__);
}

ContextHIP& ContextHIP::getInstance()
{
  static ContextHIP gpuContextHIP;
  return gpuContextHIP;
}

const DeviceProperties& ContextHIP::getDeviceProperties()
{
  int currentDeviceIndex;
  checkHIPError(hipGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  return getDeviceProperties(currentDeviceIndex);
}

const DeviceProperties& ContextHIP::getDeviceProperties(const int deviceIndex)
{
  return mDeviceProperties[deviceIndex];
}

} // namespace gpu
} // namespace its
} // namespace o2
