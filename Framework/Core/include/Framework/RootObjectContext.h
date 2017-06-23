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

#include <fairmq/FairMQMessage.h>
#include <TObject.h>

#include <vector>
#include <cassert>

namespace o2 {
namespace framework {

class RootObjectContext {
public:
  struct MessageRef {
    FairMQMessagePtr header;
    std::unique_ptr<TObject> payload;
    std::string channel;
    int index;
  };
  using Messages = std::vector<MessageRef>;

  void addObject(FairMQMessagePtr header,
                 std::unique_ptr<TObject> obj,
                 const std::string &channel,
                 int index)
  {
    mMessages.push_back(std::move(MessageRef{std::move(header),
                                             std::move(obj),
                                             channel,
                                             index}));
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
    for (auto &m : mMessages) {
      assert(m.header.get() == nullptr);
      assert(m.payload.get() != nullptr);
    }
    mMessages.clear();
  }
private:
  Messages mMessages;
};

}
}
#endif // FRAMEWORK_ROOTOBJECTCONTEXT_H
