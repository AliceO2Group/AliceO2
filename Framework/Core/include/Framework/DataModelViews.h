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
#ifndef O2_FRAMEWORK_DATASPECVIEWS_H_
#define O2_FRAMEWORK_DATASPECVIEWS_H_

#include <fairmq/FwdDecls.h>
#include <fairmq/Message.h>
#include "DomainInfoHeader.h"
#include "SourceInfoHeader.h"
#include "Headers/DataHeader.h"
#include "Framework/TimesliceSlot.h"
#include <ranges>
#include <span>

namespace o2::framework
{

struct count_payloads {
  // ends the pipeline, returns the container
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend size_t operator|(R&& r, count_payloads self)
  {
    size_t count = 0;
    size_t mi = 0;
    while (mi < r.size()) {
      auto* header = o2::header::get<o2::header::DataHeader*>(r[mi]->GetData());
      if (!header) {
        throw std::runtime_error("Not a DataHeader");
      }
      if (header->splitPayloadParts > 1 && header->splitPayloadIndex == header->splitPayloadParts) {
        count += header->splitPayloadParts;
        mi += header->splitPayloadParts + 1;
      } else {
        count += header->splitPayloadParts ? header->splitPayloadParts : 1;
        mi += header->splitPayloadParts ? 2 * header->splitPayloadParts : 2;
      }
    }
    return count;
  }
};

struct count_parts {
  // ends the pipeline, returns the number of parts
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend size_t operator|(R&& r, count_parts self)
  {
    size_t count = 0;
    size_t mi = 0;
    while (mi < r.size()) {
      auto* header = o2::header::get<o2::header::DataHeader*>(r[mi]->GetData());
      auto* sih = o2::header::get<o2::framework::SourceInfoHeader*>(r[mi]->GetData());
      auto* dih = o2::header::get<o2::framework::DomainInfoHeader*>(r[mi]->GetData());
      if (!header && !sih && !dih) {
        throw std::runtime_error("Header information not found");
      }
      // We skip oldest possible timeframe / end of stream and not consider it
      // as actual parts.
      if (dih || sih) {
        count += 1;
        mi += 2;
      } else if (header->splitPayloadParts > 1 && header->splitPayloadIndex == header->splitPayloadParts) {
        count += 1;
        mi += header->splitPayloadParts + 1;
      } else {
        count += header->splitPayloadParts ? header->splitPayloadParts : 1;
        mi += header->splitPayloadParts ? 2 * header->splitPayloadParts : 2;
      }
    }
    return count;
  }
};

struct DataRefIndices {
  size_t headerIdx;
  size_t payloadIdx;
};

struct get_pair {
  size_t pairId;
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend DataRefIndices operator|(R&& r, get_pair self)
  {
    size_t count = 0;
    size_t mi = 0;
    while (mi < r.size()) {
      auto* header = o2::header::get<o2::header::DataHeader*>(r[mi]->GetData());
      if (!header) {
        throw std::runtime_error("Not a DataHeader");
      }
      size_t diff = self.pairId - count;
      if (header->splitPayloadParts > 1 && header->splitPayloadIndex == header->splitPayloadParts) {
        // New style: one header followed by splitPayloadParts contiguous payloads.
        count += header->splitPayloadParts;
        if (self.pairId < count) {
          return {mi, mi + 1 + diff};
        }
        mi += header->splitPayloadParts + 1;
      } else if (header->splitPayloadParts > 1 && header->splitPayloadIndex != header->splitPayloadParts) {
        // Old style multi-part: splitPayloadParts [header, payload] pairs.
        // We are at the first pair of the block; jump directly.
        if (diff < header->splitPayloadParts) {
          return {mi + 2 * diff, mi + 2 * diff + 1};
        }
        count += header->splitPayloadParts;
        mi += 2 * header->splitPayloadParts;
      } else {
        // Single [header, payload] pair (splitPayloadParts == 0).
        if (self.pairId == count) {
          return {mi, mi + 1};
        }
        count += 1;
        mi += 2;
      }
    }
    throw std::runtime_error("Payload not found");
  }
};

// Advance from a DataRefIndices to the next one in O(1), reading only the
// current header.  Intended for use in iterators so that ++ is O(1) rather
// than the O(n) while-loop that get_pair requires.
//
// New-style block  (splitPayloadIndex == splitPayloadParts > 1):
//   layout: [header, payload_0, payload_1, ..., payload_{N-1}]
//   advance within block while payloads remain, then jump to the next block.
//
// Old-style block  (splitPayloadIndex != splitPayloadParts, splitPayloadParts > 1)
// or single pair   (splitPayloadParts == 0):
//   layout: [header, payload]  – always advance by two messages.
struct get_next_pair {
  DataRefIndices current;
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend DataRefIndices operator|(R&& r, get_next_pair self)
  {
    size_t hIdx = self.current.headerIdx;
    auto* header = o2::header::get<o2::header::DataHeader*>(r[hIdx]->GetData());
    if (!header) {
      throw std::runtime_error("Not a DataHeader");
    }
    if (header->splitPayloadParts > 1 && header->splitPayloadIndex == header->splitPayloadParts) {
      // New-style block: one header followed by splitPayloadParts contiguous payloads.
      if (self.current.payloadIdx < hIdx + header->splitPayloadParts) {
        // More sub-payloads remain in this block.
        return {hIdx, self.current.payloadIdx + 1};
      }
      // Last sub-payload consumed; move to the first pair of the next block.
      size_t nextHIdx = hIdx + header->splitPayloadParts + 1;
      return {nextHIdx, nextHIdx + 1};
    }
    // Old-style [header, payload] pairs or a single pair: advance by two messages.
    return {hIdx + 2, hIdx + 3};
  }
};

struct get_dataref_indices {
  size_t part;
  size_t subPart;
  // ends the pipeline, returns the number of parts
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend DataRefIndices operator|(R&& r, get_dataref_indices self)
  {
    size_t count = 0;
    size_t mi = 0;
    while (mi < r.size()) {
      auto* header = o2::header::get<o2::header::DataHeader*>(r[mi]->GetData());
      if (!header) {
        throw std::runtime_error("Not a DataHeader");
      }
      if (header->splitPayloadParts > 1 && header->splitPayloadIndex == header->splitPayloadParts) {
        if (self.part == count) {
          return {mi, mi + 1 + self.subPart};
        }
        count += 1;
        mi += header->splitPayloadParts + 1;
      } else {
        if (self.part == count) {
          return {mi, mi + self.subPart + 1};
        }
        count += 1;
        mi += 2;
      }
    }
    throw std::runtime_error("Payload not found");
  }
};

struct get_header {
  size_t id;
  // ends the pipeline, returns the number of parts
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend auto& operator|(R&& r, get_header self)
  {
    return r[(r | get_dataref_indices{self.id, 0}).headerIdx];
  }
};

struct get_payload {
  size_t part;
  size_t subPart;
  // ends the pipeline, returns the number of parts
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend auto& operator|(R&& r, get_payload self)
  {
    return r[(r | get_dataref_indices{self.part, self.subPart}).payloadIdx];
  }
};

struct get_num_payloads {
  size_t n;
  // ends the pipeline, returns the number of payloads which are associated
  // to the multipart n-th sequence of messages found in the range
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend size_t operator|(R&& r, get_num_payloads self)
  {
    size_t count = 0;
    size_t mi = 0;
    // Un
    while (mi < r.size()) {
      auto* header = o2::header::get<o2::header::DataHeader*>(r[mi]->GetData());
      if (!header) {
        throw std::runtime_error("Not a DataHeader");
      }
      if (header->splitPayloadParts > 1 && (header->splitPayloadIndex == header->splitPayloadParts)) {
        // This is the case for the new multi payload messages where the number of parts
        // is as many as the splitPayloadParts number.
        if (self.n == count) {
          return header->splitPayloadParts;
        }
        // For multipayload we skip all the parts and their associated header
        count += 1;
        mi += header->splitPayloadParts + 1;
      } else {
        // This is the case of a multipart (header, payload), (header, payload), ...
        // sequence where we know how many pairs are there.
        // When splitPayloadParts == 0, it means it is a non-multipart (header, payload)
        // pair. Each pair has exactly 1 payload.
        auto pairs = header->splitPayloadParts ? header->splitPayloadParts : 1;
        if (self.n < count + pairs) {
          return 1;
        }
        count += pairs;
        mi += 2 * pairs;
      }
    }
    return 0;
  }
};

struct inputs_for_slot {
  TimesliceSlot slot;
  template <typename R>
    requires requires(R r) { requires std::ranges::random_access_range<decltype(r.sets)>; }
  friend auto operator|(R&& r, inputs_for_slot self)
  {
    return std::span(r.sets[self.slot.index * r.inputsPerSlot]);
  }
};

struct messages_for_input {
  size_t inputIdx;
  template <typename R>
    requires std::ranges::random_access_range<R>
  friend std::span<fair::mq::MessagePtr> operator|(R&& r, messages_for_input self)
  {
    return std::span(r[self.inputIdx]);
  }
};

// FIXME: we should use special index classes in place of size_t
// FIXME: we need something to substitute a range in the store with another

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATASPECVIEWS_H_
