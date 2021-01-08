// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_ARROWCONTEXT_H_
#define O2_FRAMEWORK_ARROWCONTEXT_H_

#include "Framework/FairMQDeviceProxy.h"
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class FairMQMessage;

namespace o2::framework
{

class FairMQResizableBuffer;

/// A context which holds `std::string`s being passed around
/// useful for debug purposes and as an illustration of
/// how to add a context for a new kind of object.
class ArrowContext
{
 public:
  ArrowContext(FairMQDeviceProxy proxy)
    : mProxy{proxy}
  {
  }

  struct MessageRef {
    /// The header to be associated with the message
    std::unique_ptr<FairMQMessage> header;
    /// The actual buffer holding the ArrowData
    std::shared_ptr<FairMQResizableBuffer> buffer;
    /// The function to call to finalise the builder into the message
    std::function<void(std::shared_ptr<FairMQResizableBuffer>)> finalize;
    std::string channel;
  };

  using Messages = std::vector<MessageRef>;

  void addBuffer(std::unique_ptr<FairMQMessage> header,
                 std::shared_ptr<FairMQResizableBuffer> buffer,
                 std::function<void(std::shared_ptr<FairMQResizableBuffer>)> finalize,
                 const std::string& channel)
  {
    mMessages.push_back(std::move(MessageRef{std::move(header),
                                             std::move(buffer),
                                             std::move(finalize),
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
      //      assert(m.header.get() == nullptr);
      //      assert(m.payload.get() != nullptr);
    }
    mMessages.clear();
  }

  FairMQDeviceProxy& proxy()
  {
    return mProxy;
  }

  void updateBytesSent(size_t value)
  {
    mBytesSent += value;
  }

  void updateBytesDestroyed(size_t value)
  {
    mBytesDestroyed += value;
  }

  void updateMessagesSent(size_t value)
  {
    mMessagesCreated += value;
  }

  void updateMessagesDestroyed(size_t value)
  {
    mMessagesDestroyed += value;
  }

  size_t bytesSent()
  {
    return mBytesSent;
  }

  size_t bytesDestroyed()
  {
    return mBytesDestroyed;
  }

  size_t messagesCreated()
  {
    return mMessagesCreated;
  }

  size_t messagesDestroyed()
  {
    return mMessagesDestroyed;
  }

 private:
  FairMQDeviceProxy mProxy;
  Messages mMessages;
  size_t mBytesSent = 0;
  size_t mBytesDestroyed = 0;
  size_t mMessagesCreated = 0;
  size_t mMessagesDestroyed = 0;
  size_t mRateLimit = 0;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_ARROWCONTEXT_H_
