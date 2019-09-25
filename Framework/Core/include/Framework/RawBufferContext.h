// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   RawBufferContext.h
/// \brief  **********
/// \author Gabriele G. Fronzé <gfronze at cern.ch>
/// \date   31/07/2018

#ifndef FRAMEWORK_RAWBUFFERCONTEXT_H
#define FRAMEWORK_RAWBUFFERCONTEXT_H

#include "Framework/FairMQDeviceProxy.h"
#include "CommonUtils/BoostSerializer.h"
#include <vector>
#include <cassert>
#include <string>
#include <memory>
#include "boost/any.hpp"

class FairMQMessage;

namespace o2
{
namespace framework
{

/// A context which holds bytes streams being passed around
/// Intended to be used with boost serialization methods
class RawBufferContext
{
 public:
  RawBufferContext(FairMQDeviceProxy proxy)
    : mProxy{proxy}
  {
  }
  RawBufferContext(RawBufferContext&& other)
    : mProxy{other.mProxy}, mMessages{std::move(other.mMessages)}
  {
  }

  struct MessageRef {
    std::unique_ptr<FairMQMessage> header;
    char* payload;
    std::string channel;
    std::function<std::ostringstream()> serializeMsg;
    std::function<void()> destroyPayload;
  };

  using Messages = std::vector<MessageRef>;

  void addRawBuffer(std::unique_ptr<FairMQMessage> header,
                    char* payload,
                    std::string channel,
                    std::function<std::ostringstream()> serialize,
                    std::function<void()> destructor)
  {
    mMessages.push_back(std::move(MessageRef{std::move(header), std::move(payload), std::move(channel), std::move(serialize), std::move(destructor)}));
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
    // there because what's really sent is the copy of the raw
    // payload will be cleared by the mMessages.clear()
    for (auto& m : mMessages) {
      assert(m.header == nullptr);
      assert(m.payload != nullptr);
      m.destroyPayload();
      m.payload = nullptr;
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

#endif //FRAMEWORK_RAWBUFFERCONTEXT_H
