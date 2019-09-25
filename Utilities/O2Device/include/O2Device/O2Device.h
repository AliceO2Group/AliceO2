// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @headerfile O2Device.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#ifndef O2DEVICE_H_
#define O2DEVICE_H_

#include <FairMQDevice.h>
#include <options/FairMQProgOptions.h>
#include "O2Device/Utilities.h"
#include "Monitoring/MonitoringFactory.h"
#include <stdexcept>

namespace o2
{
namespace base
{

/// just a typedef to express the fact that it is not just a FairMQParts vector,
/// it has to follow the O2 convention of header-payload-header-payload
class O2Device : public FairMQDevice
{
 public:
  using FairMQDevice::FairMQDevice;

  ~O2Device() override = default;

  /// Monitoring instance
  std::unique_ptr<o2::monitoring::Monitoring> monitoring;

  /// Provides monitoring instance
  auto GetMonitoring() { return monitoring.get(); }

  /// Connects to a monitoring backend
  void Init() override
  {
    FairMQDevice::Init();
    static constexpr const char* MonitoringUrlKey = "monitoring-url";
    std::string monitoringUrl = GetConfig()->GetValue<std::string>(MonitoringUrlKey);
    if (!monitoringUrl.empty()) {
      monitoring->addBackend(o2::monitoring::MonitoringFactory::GetBackend(monitoringUrl));
    }
  }

 private:
};
} // namespace base
} // namespace o2
#endif /* O2DEVICE_H_ */
