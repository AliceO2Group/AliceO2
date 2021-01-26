// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DRIVERCLIENT_H_
#define O2_FRAMEWORK_DRIVERCLIENT_H_

#include "Framework/ServiceHandle.h"
#include <functional>

namespace o2::framework
{

/// A service API to communicate with the driver
class DriverClient
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  /// Report some message to the Driver
  virtual void tell(const char* msg) = 0;

  /// Act on some @a event notified by the driver
  virtual void observe(const char* event, std::function<void(char const*)> callback) = 0;
  /// Flush all pending events (if connected)
  virtual void flushPending() = 0;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DRIVERCLIENT_H_
