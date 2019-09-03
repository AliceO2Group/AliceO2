// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_SIMPLERAWDEVICESERVICE_H
#define FRAMEWORK_SIMPLERAWDEVICESERVICE_H

#include "Framework/RawDeviceService.h"

namespace o2
{
namespace framework
{

/// Fairly unsophisticated service which simply stores and return the
/// requested FairMQDevice
class SimpleRawDeviceService : public RawDeviceService
{
 public:
  SimpleRawDeviceService(FairMQDevice* device)
    : mDevice(device)
  {
  }

  FairMQDevice* device() final
  {
    return mDevice;
  }

  void setDevice(FairMQDevice* device) final
  {
    mDevice = device;
  }

 private:
  FairMQDevice* mDevice;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_SIMPLERAWDEVICESERVICE_H
