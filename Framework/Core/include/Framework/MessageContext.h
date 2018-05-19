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

#include <cassert>
#include <memory>
#include <string>
#include <vector>

class FairMQMessage;

namespace o2 {
namespace framework {

class MessageContext {
public:
  using Parts = std::vector<std::unique_ptr<FairMQMessage>>;
  struct MessageRef {
    Parts parts;
    std::string channel;
  };
  using Messages = std::vector<MessageRef>;

  void addPart(Parts &&parts, const std::string &channel) {
    assert(parts.size() == 2);
    mMessages.push_back(std::move(MessageRef{std::move(parts), channel}));
    assert(parts.size() == 0);
    assert(mMessages.back().parts.size() == 2);
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
      assert(m.parts.size() == 0);
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
