// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_TEXTDRIVERCLIENT_H_
#define O2_FRAMEWORK_TEXTDRIVERCLIENT_H_

#include "Framework/DriverClient.h"

namespace o2::framework
{

struct ServiceRegistry;
struct DeviceState;

/// A text based way of communicating with the driver.
class TextDriverClient : public DriverClient
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  TextDriverClient(ServiceRegistry& registry, DeviceState& deviceState);

  /// The text based client simply sends a message on stdout which is
  /// (potentially) captured by the driver.
  void tell(const char* msg) override;
  /// Half duplex communication
  void observe(const char* event, std::function<void(char const*)> callback) override{};
  void flushPending() override;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_TEXTDRIVERCLIENT_H_
