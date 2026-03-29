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
#ifndef FRAMEWORK_MESSAGESET_H
#define FRAMEWORK_MESSAGESET_H

#include "Framework/PartRef.h"
#include <fairmq/Message.h>
#include "Framework/DataModelViews.h"
#include <memory>
#include <vector>
#include <cassert>

namespace o2::framework
{

/// A set of inflight messages.
/// The messages are stored in a linear vector. Originally, an O2 message was
/// comprised of a header-payload pair which makes indexing of pairs in the
/// storage simple. To support O2 messages with multiple payloads in a future
/// update of the data model, a message index is needed to store position in the
/// linear storage and number of messages.
/// DPL InputRecord API is providing refs of header-payload pairs, the original
/// O2 message model. For this purpose, also the pair index is filled and can
/// be used to access header and payload associated with a pair
struct MessageSet {
  // linear storage of messages
  std::vector<fair::mq::MessagePtr> messages;

  MessageSet()
    : messages()
  {
  }

  template <typename F>
  MessageSet(F getter, size_t size)
    : messages()
  {
    for (size_t i = 0; i < size; ++i) {
      messages.emplace_back(std::move(getter(i)));
    }
  }

  MessageSet(MessageSet&& other)
    : messages(std::move(other.messages))
  {
    other.clear();
  }

  MessageSet& operator=(MessageSet&& other)
  {
    if (&other == this) {
      return *this;
    }
    messages = std::move(other.messages);
    other.clear();
    return *this;
  }

  /// clear the set
  void clear()
  {
    messages.clear();
  }
};

} // namespace o2::framework

#endif // FRAMEWORK_MESSAGESET_H
