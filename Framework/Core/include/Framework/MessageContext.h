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

#include <fairmq/FairMQParts.h>
#include <vector>
#include <cassert>
#include <string>

namespace o2 {
namespace framework {

class MessageContext {
public:
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
  void prepareForTimeslice(size_t timeslice)
  {
    // Verify that everything has been sent on clear.
    for (auto &m : mMessages) {
      assert(m.parts.Size() == 0);
    }
    mMessages.clear();
    mTimeslice = timeslice;
  }

  /// This returns the current timeslice for the context. The value of the
  /// timeslice is used to determine which downstream device will get the
  /// message in case we are doing time pipelining.
  size_t timeslice() const {
    return mTimeslice;
  }
private:
  Messages mMessages;
  size_t mTimeslice;
};

}
}
#endif // FRAMEWORK_MESSAGECONTEXT_H
