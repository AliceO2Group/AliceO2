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
#include <memory>
#include <vector>
#include <cassert>

namespace o2
{
namespace framework
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
  struct Index {
    Index(size_t p, size_t s) : position(p), size(s) {}
    size_t position = 0;
    size_t size = 0;
  };
  // linear storage of messages
  std::vector<fair::mq::MessagePtr> messages;
  // message map describes O2 messages consisting of a header message and
  // payload message(s), index describes position in the linear storage
  std::vector<Index> messageMap;
  // pair map describes all messages in one sequence of header-payload pairs and
  // where in the message index the associated header and payload can be found
  struct PairMapping {
    PairMapping(size_t partId, size_t payloadId) : partIndex(partId), payloadIndex(payloadId) {}
    // O2 message where the pair is located in
    size_t partIndex = 0;
    // payload index within the O2 message
    size_t payloadIndex = 0;
  };
  std::vector<PairMapping> pairMap;

  MessageSet()
    : messages(), messageMap(), pairMap()
  {
  }

  template <typename F>
  MessageSet(F getter, size_t size)
    : messages(), messageMap(), pairMap()
  {
    add(std::forward<F>(getter), size);
  }

  MessageSet(MessageSet&& other)
    : messages(std::move(other.messages)), messageMap(std::move(other.messageMap)), pairMap(std::move(other.pairMap))
  {
    other.clear();
  }

  MessageSet& operator=(MessageSet&& other)
  {
    if (&other == this) {
      return *this;
    }
    messages = std::move(other.messages);
    messageMap = std::move(other.messageMap);
    pairMap = std::move(other.pairMap);
    other.clear();
    return *this;
  }

  /// get number of in-flight O2 messages
  size_t size() const
  {
    return messageMap.size();
  }

  /// get number of header-payload pairs
  size_t getNumberOfPairs() const
  {
    return pairMap.size();
  }

  /// get number of payloads for an in-flight message
  size_t getNumberOfPayloads(size_t mi) const
  {
    return messageMap[mi].size;
  }

  /// clear the set
  void clear()
  {
    messages.clear();
    messageMap.clear();
    pairMap.clear();
  }

  // this is more or less legacy
  // PartRef has been earlier used to store fixed header-payload pairs
  // reset the set and store content of the part ref
  void reset(PartRef&& ref)
  {
    clear();
    add(std::move(ref));
  }

  // this is more or less legacy
  // PartRef has been earlier used to store fixed header-payload pairs
  // add  content of the part ref
  void add(PartRef&& ref)
  {
    pairMap.emplace_back(messageMap.size(), 0);
    messageMap.emplace_back(messages.size(), 1);
    messages.emplace_back(std::move(ref.header));
    messages.emplace_back(std::move(ref.payload));
  }

  /// add an O2 message
  template <typename F>
  void add(F getter, size_t size)
  {
    auto partid = messageMap.size();
    messageMap.emplace_back(messages.size(), size - 1);
    for (size_t i = 0; i < size; ++i) {
      if (i > 0) {
        pairMap.emplace_back(partid, i - 1);
      }
      messages.emplace_back(std::move(getter(i)));
    }
  }

  fair::mq::MessagePtr& header(size_t partIndex)
  {
    return messages[messageMap[partIndex].position];
  }

  fair::mq::MessagePtr& payload(size_t partIndex, size_t payloadIndex = 0)
  {
    assert(partIndex < messageMap.size());
    assert(messageMap[partIndex].position + payloadIndex + 1 < messages.size());
    return messages[messageMap[partIndex].position + payloadIndex + 1];
  }

  fair::mq::MessagePtr const& header(size_t partIndex) const
  {
    return messages[messageMap[partIndex].position];
  }

  fair::mq::MessagePtr const& payload(size_t partIndex, size_t payloadIndex = 0) const
  {
    assert(partIndex < messageMap.size());
    assert(messageMap[partIndex].position + payloadIndex + 1 < messages.size());
    return messages[messageMap[partIndex].position + payloadIndex + 1];
  }

  fair::mq::MessagePtr const& associatedHeader(size_t pos) const
  {
    return messages[messageMap[pairMap[pos].partIndex].position];
  }

  fair::mq::MessagePtr const& associatedPayload(size_t pos) const
  {
    auto partIndex = pairMap[pos].partIndex;
    auto payloadIndex = pairMap[pos].payloadIndex;
    return messages[messageMap[partIndex].position + payloadIndex + 1];
  }
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_MESSAGESET_H
