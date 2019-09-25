// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_ROOTOBJETCONTEXT_H
#define FRAMEWORK_ROOTOBJETCONTEXT_H

#include "Framework/FairMQDeviceProxy.h"
#include <vector>
#include <cassert>
#include <string>
#include <memory>

class TObject;
class FairMQMessage;

namespace o2
{
namespace framework
{

/// Holds ROOT objects which are being processed by a given
/// computation.
class RootObjectContext
{
 public:
  RootObjectContext(FairMQDeviceProxy proxy)
    : mProxy{proxy}
  {
  }

  struct MessageRef {
    std::unique_ptr<FairMQMessage> header;
    std::unique_ptr<TObject> payload;
    std::string channel;
  };

  using Messages = std::vector<MessageRef>;

  void addObject(std::unique_ptr<FairMQMessage> header,
                 std::unique_ptr<TObject> obj,
                 const std::string& channel)
  {
    mMessages.push_back(std::move(MessageRef{std::move(header),
                                             std::move(obj),
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
    // there because what's really sent is the TMessage
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
#endif // FRAMEWORK_ROOTOBJECTCONTEXT_H
