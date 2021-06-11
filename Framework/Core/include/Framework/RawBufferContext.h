// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_RAWBUFFERCONTEXT_H_
#define O2_FRAMEWORK_RAWBUFFERCONTEXT_H_

#include "Framework/FairMQDeviceProxy.h"
#include "CommonUtils/BoostSerializer.h"
#include <vector>
#include <string>
#include <memory>

class FairMQMessage;

namespace o2::framework
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
  RawBufferContext(RawBufferContext&& other);

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

 private:
  FairMQDeviceProxy mProxy;
  Messages mMessages;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_RAWBUFFERCONTEXT_H_
