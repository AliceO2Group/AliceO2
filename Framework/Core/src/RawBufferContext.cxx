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

#include "Framework/RawBufferContext.h"
#include <FairMQMessage.h>

namespace o2::framework
{

void RawBufferContext::addRawBuffer(std::unique_ptr<FairMQMessage> header,
                                    char* payload,
                                    std::string channel,
                                    std::function<std::ostringstream()> serialize,
                                    std::function<void()> destructor)
{
  mMessages.push_back(std::move(MessageRef{std::move(header), std::move(payload), std::move(channel), std::move(serialize), std::move(destructor)}));
}

void RawBufferContext::clear()
{
  // On send we move the header, but the payload remains
  // there because what's really sent is the copy of the raw
  // payload will be cleared by the mMessages.clear()
  for (auto& m : mMessages) {
    assert(m.header == nullptr);
    // NOTE: payloads can be empty so m.payload == nullptr should
    //       be an actual issue.
    assert(m.payload != nullptr);
    m.destroyPayload();
    m.payload = nullptr;
  }

  mMessages.clear();
}

RawBufferContext::RawBufferContext(RawBufferContext&& other)
  : mProxy{other.mProxy}, mMessages{std::move(other.mMessages)}
{
}

} // namespace o2::framework
