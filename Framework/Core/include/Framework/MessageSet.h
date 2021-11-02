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

/// A set of associated inflight messages.
struct MessageSet {
  std::vector<PartRef> parts;

  MessageSet()
    : parts()
  {
  }

  template <typename F>
  MessageSet(F&& getter, size_t size)
    : parts()
  {
    add(std::forward<F>(getter), size);
  }

  MessageSet(MessageSet&& other)
    : parts(std::move(other.parts))
  {
    other.clear();
  }

  MessageSet& operator=(MessageSet&& other)
  {
    if (&other == this) {
      return *this;
    }
    parts = std::move(other.parts);
    other.clear();
    return *this;
  }

  size_t size() const
  {
    return parts.size();
  }

  size_t getNumberOfPayloads(size_t part) const
  {
    // this is for upcoming change of message store
    return 1;
  }

  void clear()
  {
    parts.clear();
  }

  // this is more or less legacy
  void reset(PartRef&& ref)
  {
    clear();
    add(std::move(ref));
  }

  void add(PartRef&& ref)
  {
    parts.emplace_back(std::move(ref));
  }

  template <typename F>
  void add(F getter, size_t size)
  {
    for (size_t i = 0; i < size; ++i) {
      PartRef ref{std::move(getter(i)), std::move(getter(i + 1))};
      parts.emplace_back(std::move(ref));
      ++i;
    }
  }

  FairMQMessagePtr& header(size_t partIndex)
  {
    assert(partIndex < parts.size());
    return parts[partIndex].header;
  }

  FairMQMessagePtr& payload(size_t partIndex, size_t payloadIndex = 0)
  {
    assert(partIndex < parts.size());
    // payload index will be supported in linear message store
    assert(payloadIndex == 0);
    return parts[partIndex].payload;
  }

  FairMQMessagePtr const& header(size_t partIndex) const
  {
    assert(partIndex < parts.size());
    return parts[partIndex].header;
  }

  FairMQMessagePtr const& payload(size_t partIndex) const
  {
    assert(partIndex < parts.size());
    return parts[partIndex].payload;
  }

  PartRef& operator[](size_t index)
  {
    return parts[index];
  }

  PartRef const& operator[](size_t index) const
  {
    return parts[index];
  }

  PartRef& at(size_t index)
  {
    return parts.at(index);
  }

  PartRef const& at(size_t index) const
  {
    return parts.at(index);
  }

  decltype(auto) begin()
  {
    return parts.begin();
  }

  decltype(auto) begin() const
  {
    return parts.begin();
  }

  decltype(auto) end()
  {
    return parts.end();
  }

  decltype(auto) end() const
  {
    return parts.end();
  }
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_MESSAGESET_H
