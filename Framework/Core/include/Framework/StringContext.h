// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_STRINGCONTEXT_H
#define FRAMEWORK_STRINGCONTEXT_H

#include "Framework/FairMQDeviceProxy.h"
#include <vector>
#include <cassert>
#include <string>
#include <memory>

class FairMQMessage;

namespace o2
{
namespace framework
{

/// A context which holds `std::string`s being passed around
/// useful for debug purposes and as an illustration of
/// how to add a context for a new kind of object.
class StringContext
{
 public:
  StringContext(FairMQDeviceProxy proxy)
    : mProxy{proxy}
  {
  }

  struct MessageRef {
    std::unique_ptr<FairMQMessage> header;
    std::unique_ptr<std::string> payload;
    std::string channel;
  };

  using Messages = std::vector<MessageRef>;

  void addString(std::unique_ptr<FairMQMessage> header,
                 std::unique_ptr<std::string> s,
                 const std::string& channel)
  {
    mMessages.push_back(std::move(MessageRef{std::move(header),
                                             std::move(s),
                                             channel}));
  }

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

  void clear()
  {
    // On send we move the header, but the payload remains
    // there because what's really sent is the copy of the string
    // payload will be cleared by the mMessages.clear()
    for (auto& m : mMessages) {
      assert(m.header.get() == nullptr);
      assert(m.payload.get() != nullptr);
    }
    mMessages.clear();
  }

  FairMQDeviceProxy& proxy()
  {
    return mProxy;
  }

 private:
  FairMQDeviceProxy mProxy;
  Messages mMessages;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_STRINGCONTEXT_H
