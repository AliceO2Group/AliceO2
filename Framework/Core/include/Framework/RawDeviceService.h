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
#ifndef O2_FRAMEWORK_RAWDEVICESERVICE_H_
#define O2_FRAMEWORK_RAWDEVICESERVICE_H_

#include "Framework/ServiceHandle.h"

#include <fairmq/FwdDecls.h>

namespace o2::framework
{
class DeviceSpec;

/// This service provides a hook into the actual fairmq device running the
/// computation, and allows an advanced user to modify its behavior
/// from within a workflow class. This should be used to implement special
/// `DataProcessors` like one that acts as a gateway to standard FairMQ
/// devices.
class RawDeviceService
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  virtual fair::mq::Device* device() = 0;
  virtual void setDevice(fair::mq::Device* device) = 0;
  virtual DeviceSpec const& spec() const = 0;
  /// Expose fair::mq::Device::WaitFor method to avoid having to include
  /// <fairmq/Device.h>.
  ///
  ///  @a time in millisecond to sleep
  virtual void waitFor(unsigned int time) = 0;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_RAWDEVICESERVICE_H_
