// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_MESSAGECONTEXT_H
#define FRAMEWORK_MESSAGECONTEXT_H

#include <fairmq/FairMQParts.h>
#include <vector>
#include <cassert>

namespace o2 {
namespace framework {

class MessageContext {
public:
  struct MessageRef {
    FairMQParts parts;
    std::string channel;
    int index;
  };
  using Messages = std::vector<MessageRef>;

  void addPart(FairMQParts &&parts, const std::string &channel, int index) {
    assert(parts.Size() == 2);
    mMessages.push_back(std::move(MessageRef{std::move(parts), channel, index}));
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

  void clear()
  {
    // Verify that everything has been sent on clear.
    for (auto &m : mMessages) {
      assert(m.parts.Size() == 0);
    }
    mMessages.clear();
  }
private:
  Messages mMessages;
};

}
}
#endif // FRAMEWORK_MESSAGECONTEXT_H
