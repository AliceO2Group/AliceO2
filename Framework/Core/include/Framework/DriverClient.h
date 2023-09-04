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
#ifndef O2_FRAMEWORK_DRIVERCLIENT_H_
#define O2_FRAMEWORK_DRIVERCLIENT_H_

#include "Framework/ServiceHandle.h"
#include "Framework/ServiceRegistryRef.h"
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace o2::framework
{

struct DriverEventMatcher {
  std::string prefix;
  std::function<void(std::string_view)> callback;
};

/// A service API to communicate with the driver
class DriverClient
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  /// Report some message to the Driver
  /// @a msg the message to be sent.
  /// @a size size of the message to be sent.
  /// @a flush whether the message should be flushed immediately,
  /// if possible.
  virtual void tell(char const* msg, size_t s, bool flush = true) = 0;
  void tell(std::string_view const& msg, bool flush = true)
  {
    tell(msg.data(), msg.size(), flush);
  };

  /// Request action on some @a eventType notified by the driver
  void observe(char const* eventType, std::function<void(std::string_view)> callback);

  /// Dispatch an event
  void dispatch(std::string_view event);

  /// Flush all pending events (if connected)
  /// Note: this must be guaranteed to be called from the main thread,
  /// because the libuv backend cannot queue write operations on a thread
  /// which is not the main one. In order to do this, one should
  /// have assert(mainTreadRef.isMainThread()) in the implementation.
  /// Notice also that if you want to use uv_async_send, you must make
  /// sure that you do not rely on having as many events as the times
  /// you called uv_async_send, because this is not guaranteed.
  virtual void flushPending(ServiceRegistryRef mainThreadRef) = 0;

 private:
  std::vector<DriverEventMatcher> mEventMatchers;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DRIVERCLIENT_H_
