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
/// \file ContextHIP.h
/// \brief
///

#ifndef O2_ITS_TRACKING_INCLUDE_CONTEXT_HIP_H_
#define O2_ITS_TRACKING_INCLUDE_CONTEXT_HIP_H_

#include <string>
#include <vector>
// #include "ITStracking/Definitions.h"
#include <hip/hip_runtime_api.h>
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{
namespace GPU
{

struct DeviceProperties final {
  std::string name;
  int gpuProcessors;
  int streamProcessors;
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

class ContextHIP final
{
 public:
  static ContextHIP& getInstance();

  ContextHIP(const ContextHIP&);
  ContextHIP& operator=(const ContextHIP&);

  const DeviceProperties& getDeviceProperties();
  const DeviceProperties& getDeviceProperties(const int);

 private:
  ContextHIP(bool dumpDevices = true);
  ~ContextHIP() = default;

  int mDevicesNum;
  std::vector<DeviceProperties> mDeviceProperties;
};
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_CONTEXT_HIP_H_ */
