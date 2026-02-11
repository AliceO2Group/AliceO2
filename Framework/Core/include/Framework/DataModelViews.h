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
#include <ranges>

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
        count += header->splitPayloadParts;
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
        count += header->splitPayloadParts;
        if (self.pairId < count) {
          return {mi, mi + 1 + diff};
        }
        mi += header->splitPayloadParts + 1;
      } else {
        count += header->splitPayloadParts ? header->splitPayloadParts : 1;
        if (self.pairId < count) {
          return {mi, mi + 2 * diff + 1};
        }
        mi += header->splitPayloadParts ? 2 * header->splitPayloadParts : 2;
      }
    }
    throw std::runtime_error("Payload not found");
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
          return {mi, mi + 2 * self.subPart + 1};
        }
        count += 1;
        mi += header->splitPayloadParts ? 2 * header->splitPayloadParts : 2;
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
  friend fair::mq::MessagePtr& operator|(R&& r, get_header self)
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
  friend fair::mq::MessagePtr& operator|(R&& r, get_payload self)
  {
    return r[(r | get_dataref_indices{self.part, self.subPart}).payloadIdx];
  }
};

struct get_num_payloads {
  size_t id;
  // ends the pipeline, returns the number of parts
  template <typename R>
    requires std::ranges::random_access_range<R> && std::ranges::sized_range<R>
  friend size_t operator|(R&& r, get_num_payloads self)
  {
    size_t count = 0;
    size_t mi = 0;
    while (mi < r.size()) {
      auto* header = o2::header::get<o2::header::DataHeader*>(r[mi]->GetData());
      if (!header) {
        throw std::runtime_error("Not a DataHeader");
      }
      if (self.id == count) {
        if (header->splitPayloadParts > 1 && (header->splitPayloadIndex == header->splitPayloadParts)) {
          return header->splitPayloadParts;
        } else {
          return 1;
        }
      }
      if (header->splitPayloadParts > 1 && (header->splitPayloadIndex == header->splitPayloadParts)) {
        count += 1;
        mi += header->splitPayloadParts + 1;
      } else {
        count += 1;
        mi += header->splitPayloadParts ? 2 * header->splitPayloadParts : 2;
      }
    }
    return 0;
  }
};

struct MessageSet;

struct MessageStore {
  std::span<MessageSet> sets;
  size_t inputsPerSlot = 0;
};

struct inputs_for_slot {
  TimesliceSlot slot;
  template <typename R>
    requires requires(R r) { std::ranges::random_access_range<decltype(r.sets)>; }
  friend std::span<o2::framework::MessageSet> operator|(R&& r, inputs_for_slot self)
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
    return r[self.inputIdx].messages;
  }
};

// FIXME: we should use special index classes in place of size_t
// FIXME: we need something to substitute a range in the store with another

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATASPECVIEWS_H_
