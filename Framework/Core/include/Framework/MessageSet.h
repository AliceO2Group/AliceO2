// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_MESSAGESET_H_
#define O2_FRAMEWORK_MESSAGESET_H_

#include "Framework/PartRef.h"
#include <Message.h>
#include <fairmq/FwdDecls.h>
#include <memory>
#include <vector>
#include <cassert>
#include <concepts>

namespace o2::framework
{

template <typename T>
concept MessageSetFiller = requires(T t, size_t n) {
  { t(n) } -> std::same_as<fair::mq::MessagePtr>;
};

template <typename T>
concept MessageSetCounter = requires(fair::mq::MessagePtr && (*t)(fair::mq::MessagePtr&&), fair::mq::MessagePtr&& ref) {
  { t(std::forward<fair::mq::MessagePtr>(ref)) } -> std::same_as<fair::mq::MessagePtr&&>;
};

template <typename T>
concept MessageSetDisposer = requires(void (*t)(fair::mq::MessagePtr&&), fair::mq::MessagePtr&& ref) {
  { t(std::forward<fair::mq::MessagePtr>(ref)) } -> std::same_as<void>;
};

// Sometimes we fill from a PartRef, e.g. (header, payload pair)
// So we need some special code for it.
template <typename T>
concept PartRefFiller = requires(T t, size_t n) {
  { t(n) } -> std::same_as<PartRef>;
};

template <typename T>
concept PartRefCounter = requires(PartRef && (*t)(PartRef&&), PartRef&& ref) {
  { t(std::forward<PartRef>(ref)) } -> std::same_as<PartRef&&>;
};

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
  static auto passthrough(fair::mq::MessagePtr&& ref) -> fair::mq::MessagePtr&& { return std::forward<fair::mq::MessagePtr>(ref); }
  static auto passthrough_partref(o2::framework::PartRef&& ref) -> o2::framework::PartRef&& { return std::forward<o2::framework::PartRef>(ref); }
  // Use this when you want to delete the messages on clear.
  static auto destroy_message(fair::mq::MessagePtr&& ref) -> void
  {
    fair::mq::MessagePtr toDelete(nullptr);
    ref.swap(toDelete);
  }
  static auto noop(fair::mq::MessagePtr&& ref) -> void {}
  static auto assert_empty(fair::mq::MessagePtr&& ref) -> void { assert(ref.get() == nullptr); }
  static auto enforce_empty(fair::mq::MessagePtr&& ref) -> void;

  struct Index {
    size_t position = 0;
    size_t size = 0;
  };
  // linear storage of messages
  std::vector<std::unique_ptr<fair::mq::Message>> messages;
  // message map describes O2 messages consisting of a header message and
  // payload message(s), index describes position in the linear storage
  std::vector<Index> messageMap;
  // pair map describes all messages in one sequence of header-payload pairs and
  // where in the message index the associated header and payload can be found
  struct PairMapping {
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

  // Allow creating a message set via a getter.
  // The counting function will be invoked only
  // once per message. If you want to augment a
  // MessageSet use the merge method.
  MessageSet(MessageSetFiller auto getter, size_t size, MessageSetCounter auto counter)
    : messages(), messageMap(), pairMap()
  {
    messages.reserve(size);
    pairMap.reserve(size - 1);
    messageMap.emplace_back(Index{.position = 0, .size = size - 1});
    for (size_t i = 0; i < size; ++i) {
      if (i > 0) {
        pairMap.emplace_back(0, i - 1);
      }
      messages.emplace_back(std::move(counter(getter(i))));
    }
  }

  MessageSet(PartRefFiller auto filler, size_t nPartRef, PartRefCounter auto counter)
    : messages(), messageMap(), pairMap()
  {
    messages.reserve(2 * nPartRef);    // Beacause messages contains all the messages
    messageMap.reserve(nPartRef);      // Because the message map tracks how many (header, payload0, payload1 ...) there are
    pairMap.reserve(2 * nPartRef - 1); // Because pairMap tracks (header, payload0), (header, payload1), etc.

    for (size_t i = 0; i < nPartRef; ++i) {
      pairMap.emplace_back(PairMapping{.partIndex = messageMap.size(), .payloadIndex = 0}); // Because this is the first one
      messageMap.emplace_back(Index{.position = i, .size = 1});                             // Because a PartRef only has 2 messages (and 1 payload)
      o2::framework::PartRef ref = counter(filler(i));
      messages.emplace_back(std::move(ref.header));
      messages.emplace_back(std::move(ref.payload));
    }
  }

  MessageSet(MessageSet&& other)
    : messages(std::move(other.messages)), messageMap(std::move(other.messageMap)), pairMap(std::move(other.pairMap))
  {
    other.clear(MessageSet::noop);
  }

  MessageSet& operator=(MessageSet&& other)
  {
    if (&other == this) {
      return *this;
    }
    messages = std::move(other.messages);
    messageMap = std::move(other.messageMap);
    pairMap = std::move(other.pairMap);
    other.clear(MessageSet::noop);
    return *this;
  }

  ~MessageSet();

  /// get number of in-flight O2 messages
  [[nodiscard]] size_t size() const
  {
    return messageMap.size();
  }

  /// get number of header-payload pairs
  [[nodiscard]] size_t getNumberOfPairs() const
  {
    return pairMap.size();
  }

  /// get number of payloads for an in-flight message
  [[nodiscard]] size_t getNumberOfPayloads(size_t mi) const
  {
    return messageMap[mi].size;
  }

  /// clear the set
  void clear(MessageSetDisposer auto dispose)
  {
    for (auto& message : messages) {
      dispose(std::move(message));
    }
    messages.clear();
    messageMap.clear();
    pairMap.clear();
  }

  /// Add messages in bulk. We are guaranteed that this
  /// function is executed only once for each incoming message
  /// so it can be used to trigger the early forwarding.
  ///
  void merge(MessageSet&& other)
  {
    auto partid = messageMap.size();
    messageMap.emplace_back(messages.size(), other.messages.size() - 1);
    for (size_t i = 0; i < other.messages.size(); ++i) {
      if (i > 0) {
        pairMap.emplace_back(partid, i - 1);
      }
      messages.emplace_back(std::move(other.messages[i]));
    }
    // Every message should be removed once the MessageSet is
    // merged.
    other.clear(MessageSet::assert_empty);
  }

  // This should really be used to give ownership to something else.
  [[nodiscard]] std::unique_ptr<fair::mq::Message> extractHeader(size_t partIndex)
  {
    return std::move(messages[messageMap[partIndex].position]);
  }

  [[nodiscard]] std::unique_ptr<fair::mq::Message> extractPayload(size_t partIndex, size_t payloadIndex = 0)
  {
    assert(partIndex < messageMap.size());
    assert(messageMap[partIndex].position + payloadIndex + 1 < messages.size());
    return std::move(messages[messageMap[partIndex].position + payloadIndex + 1]);
  }

  [[nodiscard]] std::unique_ptr<fair::mq::Message> const& header(size_t partIndex) const
  {
    return messages[messageMap[partIndex].position];
  }

  [[nodiscard]] std::unique_ptr<fair::mq::Message> const& payload(size_t partIndex, size_t payloadIndex = 0) const
  {
    assert(partIndex < messageMap.size());
    assert(messageMap[partIndex].position + payloadIndex + 1 < messages.size());
    return messages[messageMap[partIndex].position + payloadIndex + 1];
  }

  [[nodiscard]] std::unique_ptr<fair::mq::Message> const& associatedHeader(size_t pos) const
  {
    return messages[messageMap[pairMap[pos].partIndex].position];
  }

  [[nodiscard]] std::unique_ptr<fair::mq::Message> const& associatedPayload(size_t pos) const
  {
    auto partIndex = pairMap[pos].partIndex;
    auto payloadIndex = pairMap[pos].payloadIndex;
    return messages[messageMap[partIndex].position + payloadIndex + 1];
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_MESSAGESET_H_
