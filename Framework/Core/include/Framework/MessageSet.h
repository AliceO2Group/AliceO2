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
#ifndef O2_FRAMEWORK_MESSAGESET_H_
#define O2_FRAMEWORK_MESSAGESET_H_

#include "Framework/PartRef.h"
#include <memory>
#include <vector>
#include <cassert>

namespace o2::framework
{

/// A set of associated inflight messages.
struct MessageSet {
  struct Index {
    Index(size_t p, size_t s) : position(p), size(s) {}
    size_t position = 0;
    size_t size = 0;
  };
  std::vector<std::unique_ptr<FairMQMessage>> messages;
  std::vector<Index> index;

  MessageSet()
    : messages(), index()
  {
  }

  template <typename F>
  MessageSet(F getter, size_t size)
    : messages(), index()
  {
    add(std::forward<F>(getter), size);
  }

  MessageSet(MessageSet&& other)
    : messages(std::move(other.messages)), index(std::move(other.index))
  {
    other.clear();
  }

  MessageSet& operator=(MessageSet&& other)
  {
    if (&other == this) {
      return *this;
    }
    messages = std::move(other.messages);
    index = std::move(other.index);
    other.clear();
    return *this;
  }

  size_t size() const
  {
    return index.size();
  }

  size_t getNumberOfPayloads(size_t part) const
  {
    return index[part].size;
  }

  void clear()
  {
    messages.clear();
    index.clear();
  }

  // this is more or less legacy
  void reset(PartRef&& ref)
  {
    clear();
    add(std::move(ref));
  }

  void add(PartRef&& ref)
  {
    index.emplace_back(messages.size(), 1);
    messages.emplace_back(std::move(ref.header));
    messages.emplace_back(std::move(ref.payload));
  }

  template <typename F>
  void add(F getter, size_t size)
  {
    index.emplace_back(messages.size(), size - 1);
    for (size_t i = 0; i < size; ++i) {
      messages.emplace_back(std::move(getter(i)));
    }
  }

  std::unique_ptr<FairMQMessage>& header(size_t partIndex)
  {
    return messages[index[partIndex].position];
  }

  std::unique_ptr<FairMQMessage>& payload(size_t partIndex, size_t payloadIndex = 0)
  {
    assert(partIndex < index.size());
    assert(index[partIndex].position + payloadIndex + 1 < messages.size());
    return messages[index[partIndex].position + payloadIndex + 1];
  }

  std::unique_ptr<FairMQMessage> const& header(size_t partIndex) const
  {
    return messages[index[partIndex].position];
  }

  std::unique_ptr<FairMQMessage> const& payload(size_t partIndex, size_t payloadIndex = 0) const
  {
    assert(partIndex < index.size());
    assert(index[partIndex].position + payloadIndex + 1 < messages.size());
    return messages[index[partIndex].position + payloadIndex + 1];
  }
};

} // namespace o2
#endif // O2_FRAMEWORK_MESSAGESET_H_
