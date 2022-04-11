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
#ifndef O2_FRAMEWORK_SIMPLERAWDEVICESERVICE_H_
#define O2_FRAMEWORK_SIMPLERAWDEVICESERVICE_H_

#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"

namespace o2::framework
{

/// Fairly unsophisticated service which simply stores and returns the
/// requested fair::mq::Device and DeviceSpec
class SimpleRawDeviceService : public RawDeviceService
{
 public:
  SimpleRawDeviceService(fair::mq::Device* device, DeviceSpec const& spec)
    : mDevice(device), mSpec(spec)
  {
  }

  fair::mq::Device* device() final
  {
    return mDevice;
  }

  void setDevice(fair::mq::Device* device) final
  {
    mDevice = device;
  }

  DeviceSpec const& spec() const final
  {
    return mSpec;
  }

  void waitFor(unsigned int ms) final;

 private:
  fair::mq::Device* mDevice;
  DeviceSpec const& mSpec;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_SIMPLERAWDEVICESERVICE_H__
