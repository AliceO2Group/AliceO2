// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_WSDRIVERCLIENT_H_
#define O2_FRAMEWORK_WSDRIVERCLIENT_H_

#include "Framework/DriverClient.h"
#include <memory>
#include <string>
#include <vector>

typedef struct uv_connect_s uv_connect_t;

namespace o2::framework
{

struct ServiceRegistry;
struct DeviceState;
struct WSDPLClient;
struct DeviceSpec;

/// Communicate between driver and devices via a websocket
/// This implementation is enabled if you pass --driver-client ws://
/// as an option.
class WSDriverClient : public DriverClient
{
 public:
  WSDriverClient(ServiceRegistry& registry, DeviceState& state, char const* ip, unsigned short port);
  void tell(const char* msg) override;
  void observe(const char* command, std::function<void(char const*)>) override;
  void flushPending() override;
  void setDPLClient(std::unique_ptr<WSDPLClient>);
  void setConnection(uv_connect_t* connection) { mConnection = connection; };
  DeviceSpec const& spec() { return mSpec; }
  // Initiate a websocket session
  void sendHandshake();

 private:
  DeviceSpec const& mSpec;
  bool mConnected = false;
  std::vector<std::string> mBacklog;
  uv_connect_t* mConnection = nullptr;
  std::unique_ptr<WSDPLClient> mClient;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_WSDRIVERCLIENT_H_
