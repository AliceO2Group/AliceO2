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
#ifndef O2_FRAMEWORK_TEXTDRIVERCLIENT_H_
#define O2_FRAMEWORK_TEXTDRIVERCLIENT_H_

#include "Framework/DriverClient.h"
#include "Framework/ServiceRegistryRef.h"

namespace o2::framework
{

struct ServiceRegistry;
struct DeviceState;

/// A text based way of communicating with the driver.
class TextDriverClient : public DriverClient
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  TextDriverClient(ServiceRegistryRef registry, DeviceState& deviceState);

  /// The text based client simply sends a message on stdout which is
  /// (potentially) captured by the driver.
  void tell(char const* msg, size_t s, bool flush = true) final;
  void flushPending() final;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_TEXTDRIVERCLIENT_H_
