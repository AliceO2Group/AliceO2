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

#ifndef O2_FRAMEWORK_RAWBUFFERCONTEXT_H_
#define O2_FRAMEWORK_RAWBUFFERCONTEXT_H_

#include "Framework/FairMQDeviceProxy.h"
#include "Framework/RoutingIndices.h"
#include <vector>
#include <string>
#include <memory>

#include <fairmq/FwdDecls.h>

namespace o2::framework
{

/// A context which holds bytes streams being passed around
/// Intended to be used with boost serialization methods
class RawBufferContext
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Stream;

  RawBufferContext(FairMQDeviceProxy& proxy)
    : mProxy{proxy}
  {
  }
  RawBufferContext(RawBufferContext&& other);

  struct MessageRef {
    std::unique_ptr<fair::mq::Message> header;
    char* payload;
    RouteIndex routeIndex;
    std::function<std::ostringstream()> serializeMsg;
    std::function<void()> destroyPayload;
  };

  using Messages = std::vector<MessageRef>;

  void addRawBuffer(std::unique_ptr<fair::mq::Message> header,
                    char* payload,
                    RouteIndex routeIndex,
                    std::function<std::ostringstream()> serialize,
                    std::function<void()> destructor);

  Messages::iterator begin()
  {
    return mMessages.begin();
  }

  Messages::iterator end()
  {
    return mMessages.end();
  }

  size_t size()
  {
    return mMessages.size();
  }

  void clear();

  FairMQDeviceProxy& proxy()
  {
    return mProxy;
  }

  int countDeviceOutputs(bool excludeDPLOrigin);

 private:
  FairMQDeviceProxy& mProxy;
  Messages mMessages;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_RAWBUFFERCONTEXT_H_
