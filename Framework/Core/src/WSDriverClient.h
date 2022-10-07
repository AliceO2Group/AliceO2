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
#ifndef O2_FRAMEWORK_WSDRIVERCLIENT_H_
#define O2_FRAMEWORK_WSDRIVERCLIENT_H_

#include "Framework/DriverClient.h"
#include "Framework/ServiceRegistryRef.h"
#include <uv.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

typedef struct uv_connect_s uv_connect_t;
typedef struct uv_async_s uv_async_t;

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
  WSDriverClient(ServiceRegistryRef registry, DeviceState& state, char const* ip, unsigned short port);
  ~WSDriverClient();
  void tell(const char* msg, size_t s, bool flush = true) final;
  void flushPending() final;
  void setDPLClient(std::unique_ptr<WSDPLClient>);
  void setConnection(uv_connect_t* connection) { mConnection = connection; };
  DeviceSpec const& spec() { return mSpec; }
  // Initiate a websocket session
  void sendHandshake();
  std::mutex& mutex() { return mClientMutex; }

 private:
  /// Use this to awake the main thread.
  void awake();
  // Whether or not we managed to connect.
  std::atomic<bool> mConnected = false;
  std::mutex mClientMutex;
  DeviceSpec const& mSpec;
  std::vector<uv_buf_t> mBacklog;
  uv_async_t* mAwakeMainThread = nullptr;
  uv_connect_t* mConnection = nullptr;
  std::unique_ptr<WSDPLClient> mClient;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_WSDRIVERCLIENT_H_
