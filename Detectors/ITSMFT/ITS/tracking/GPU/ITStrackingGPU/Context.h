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
/// \file Context.h
/// \brief
///

#ifndef ITSTRACKINGGPU_CONTEXT_H_
#define ITSTRACKINGGPU_CONTEXT_H_

#include <string>
#include <vector>
#include "ITStracking/Definitions.h"

namespace o2
{
namespace its
{
namespace gpu
{

struct DeviceProperties final {
  std::string name;
  int gpuProcessors;
  int gpuCores;
  long globalMemorySize;
  long constantMemorySize;
  long sharedMemorySize;
  long maxClockRate;
  int busWidth;
  long l2CacheSize;
  long registersPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int maxBlocksPerSM;
  dim3 maxThreadsDim;
  dim3 maxGridDim;
};

class Context final
{
 public:
  static Context& getInstance();

  Context(const Context&);
  Context& operator=(const Context&);

  const DeviceProperties& getDeviceProperties();
  const DeviceProperties& getDeviceProperties(const int);

 private:
  Context(bool dumpDevices = false);
  ~Context() = default;

  int mDevicesNum;
  std::vector<DeviceProperties> mDeviceProperties;
};
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_ */
