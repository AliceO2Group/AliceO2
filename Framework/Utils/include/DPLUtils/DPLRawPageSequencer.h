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
#ifndef FRAMEWORK_UTILS_DPLRAWPAGESEQUENCER_H
#define FRAMEWORK_UTILS_DPLRAWPAGESEQUENCER_H

/// @file   DPLRawPageSequencer.h
/// @author Matthias Richter
/// @since  2021-07-09
/// @brief  A parser and sequencer utility for raw pages within DPL input

#include "DPLUtils/RawParser.h"
#include "Framework/DataRef.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/InputRecordWalker.h"
#include <utility> // std::declval

namespace o2::framework
{
class InputRecord;

/// @class DPLRawPageSequencer
/// @brief This utility handles transparently the DPL inputs and triggers
/// a customizable action on sequences of consecutive raw pages following
/// similar RDH features, e.g. the same FEE ID.
///
/// A DPL processor will receive raw pages accumulated on three levels:
///   1) the DPL processor has one or more input route(s)
///   2) multiple parts per input route (split payloads or multiple input
///      specs matching the same route spec
///   3) variable number of raw pages in one payload
///
/// The DPLRawPageSequencer loops transparently over all inputs matching
/// the optional filter, and partitions input buffers into sequences of
/// raw pages matching the provided predicate by binary search.
///
/// Note: binary search requires that all raw pages must have a fixed
/// length, only the last page can be shorter.
///
/// Usage:
///   auto isSameRdh = [](const char* left, const char* right) -> bool {
///     // implement the condition here
///     return left == right;
///   };
///   std::vector<std::pair<const char*, size_t>> pages;
///   auto insertPages = [&pages](const char* ptr, size_t n) -> void {
///     // as an example, the sequences are simply stored in a vector
///     pages.emplace_back(ptr, n);
///   };
///   DPLRawPageSequencer(inputs)(isSameRdh, insertPages);
///
/// TODO:
///   - support configurable page length
class DPLRawPageSequencer
{
 public:
  using rawparser_type = RawParser<8192>;
  using buffer_type = typename rawparser_type::buffer_type;

  DPLRawPageSequencer() = delete;
  DPLRawPageSequencer(InputRecord& inputs, std::vector<InputSpec> filterSpecs = {}) : mInput(inputs, filterSpecs) {}

  template <typename Predicate, typename Inserter>
  void operator()(Predicate&& pred, Inserter&& inserter)
  {
    return binary(std::forward<Predicate>(pred), std::forward<Inserter>(inserter));
  }

  template <typename Predicate, typename Inserter>
  void binary(Predicate pred, Inserter inserter)
  {
    for (auto const& ref : mInput) {
      auto size = DataRefUtils::getPayloadSize(ref);
      auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto const pageSize = rawparser_type::max_size;
      auto nPages = size / pageSize + (size % pageSize ? 1 : 0);
      if (nPages == 0) {
        continue;
      }
      // FIXME: automatic type from inserter/predicate?
      const char* iterator = ref.payload;

      auto check = [&pred, &pageSize, payload = ref.payload](size_t left, size_t right) -> bool {
        return pred(payload + left * pageSize, payload + right * pageSize);
      };
      auto insert = [&inserter, &pageSize, payload = ref.payload](size_t pos, size_t n, uint32_t subSpec) -> void {
        inserter(payload + pos * pageSize, n, subSpec);
      };
      // binary search the next different page based on the check predicate
      auto search = [&check](size_t first, size_t n) -> size_t {
        auto count = n;
        auto pos = first;
        while (count > 0) {
          auto step = count / 2;
          if (check(first, pos + step)) {
            // still the same
            pos += step;
            count = n - (pos - first);
          } else {
            if (step == 1) {
              pos += step;
              break;
            }
            count = step;
          }
        }
        return pos;
      };

      size_t p = 0;
      do {
        // insert the full block if the last RDH matches the position
        if (check(p, nPages - 1)) {
          insert(p, nPages - p, dh->subSpecification);
          break;
        }
        auto q = search(p, nPages - p);
        insert(p, q - p, dh->subSpecification);
        p = q;
      } while (p < nPages);
      // if payloads are consecutive in memory we could apply this algorithm even over
      // O2 message boundaries
    }
  }

  template <typename Predicate, typename Inserter>
  void forward(Predicate check, Inserter inserter)
  {
    for (auto const& ref : mInput) {
      auto size = DataRefUtils::getPayloadSize(ref);
      o2::framework::RawParser parser(ref.payload, size);
      auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const char* ptr = nullptr;
      int count = 0;
      for (auto it = parser.begin(); it != parser.end(); it++) {
        const char* current = reinterpret_cast<const char*>(it.raw());
        if (ptr == nullptr) {
          ptr = current;
        } else if (check(ptr, current) == false) {
          if (count) {
            inserter(ptr, count, dh->subSpecification);
          }
          count = 0;
          ptr = current;
        }
        count++;
      }
      if (count) {
        inserter(ptr, count, dh->subSpecification);
      }
    }
  }

 private:
  InputRecordWalker mInput;
};

} // namespace o2::framework

#endif //FRAMEWORK_UTILS_DPLRAWPAGESEQUENCER_H
