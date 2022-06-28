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
#ifndef O2_FRAMEWORK_STRINGCONTEXT_H_
#define O2_FRAMEWORK_STRINGCONTEXT_H_

#include "Framework/FairMQDeviceProxy.h"
#include "Framework/RouteState.h"
#include "Framework/RoutingIndices.h"
#include <vector>
#include <string>
#include <memory>

#include <fairmq/FwdDecls.h>

namespace o2::framework
{

/// A context which holds `std::string`s being passed around
/// useful for debug purposes and as an illustration of
/// how to add a context for a new kind of object.
class StringContext
{
 public:
  StringContext(FairMQDeviceProxy& proxy)
    : mProxy(proxy)
  {
  }

  struct MessageRef {
    std::unique_ptr<fair::mq::Message> header;
    std::unique_ptr<std::string> payload;
    RouteIndex routeIndex;
  };

  using Messages = std::vector<MessageRef>;

  void addString(std::unique_ptr<fair::mq::Message> header,
                 std::unique_ptr<std::string> s,
                 RouteIndex routeIndex);

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

 private:
  FairMQDeviceProxy& mProxy;
  Messages mMessages;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_STRINGCONTEXT_H_
