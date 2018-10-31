// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_MESSAGECONTEXT_H
#define FRAMEWORK_MESSAGECONTEXT_H

#include "Framework/ContextRegistry.h"
#include "Framework/FairMQDeviceProxy.h"

#include <fairmq/FairMQParts.h>

#include <vector>
#include <cassert>
#include <string>

class FairMQDevice;

namespace o2
{
namespace framework
{

class MessageContext {
public:
 MessageContext(FairMQDeviceProxy proxy)
   : mProxy{ proxy }
 {
 }

 struct MessageRef {
   FairMQParts parts;
   std::string channel;
  };
  using Messages = std::vector<MessageRef>;

  void addPart(FairMQParts &&parts, const std::string &channel) {
    assert(parts.Size() == 2);
    mMessages.push_back(std::move(MessageRef{std::move(parts), channel}));
    assert(parts.Size() == 0);
    assert(mMessages.back().parts.Size() == 2);
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

  /// Prepares the context to create messages for the given timeslice. This
  /// expects that the previous context was already sent and can be completely
  /// discarded.
  void clear()
  {
    // Verify that everything has been sent on clear.
    for (auto &m : mMessages) {
      assert(m.parts.Size() == 0);
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

/// Helper to get the context from the registry.
template <>
inline MessageContext*
  ContextRegistry::get<MessageContext>()
{
  return reinterpret_cast<MessageContext*>(mContextes[0]);
}

/// Helper to set the context from the registry.
template <>
inline void
  ContextRegistry::set<MessageContext>(MessageContext* context)
{
  mContextes[0] = context;
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_MESSAGECONTEXT_H
