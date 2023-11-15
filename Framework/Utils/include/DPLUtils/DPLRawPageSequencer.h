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

// Framework does not depend on detectors, but this file is header-only.
// So we just include the RDH header, and the user must make sure it is
// available. Otherwise there is no way to properly parse raw data
// without having access to RDH info
#include "DetectorsRaw/RDHUtils.h"

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

  template <typename Predicate, typename Inserter, typename Precheck>
  int operator()(Predicate&& pred, Inserter&& inserter, Precheck preCheck)
  {
    return binary(std::forward<Predicate>(pred), std::forward<Inserter>(inserter), std::forward<Precheck>(preCheck));
  }

  template <typename Predicate, typename Inserter>
  int operator()(Predicate&& pred, Inserter&& inserter)
  {
    return binary(std::forward<Predicate>(pred), std::forward<Inserter>(inserter), [](...) { return true; });
  }

  template <typename Predicate, typename Inserter>
  int binary(Predicate pred, Inserter inserter)
  {
    return binary(std::forward<Predicate>(pred), std::forward<Inserter>(inserter), [](...) { return true; });
  }

  template <typename Predicate, typename Inserter, typename Precheck>
  int binary(Predicate pred, Inserter inserter, Precheck preCheck)
  {
    int retVal = 0;
    for (auto const& ref : mInput) {
      auto size = DataRefUtils::getPayloadSize(ref);
      const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (dh == nullptr) {
        continue;
      }
      if (size == 0) {
        if (dh->subSpecification == 0xDEADBEEF) {
          raw_parser::RawParserHelper::warnDeadBeef(dh);
        }
        continue;
      }
      auto const pageSize = rawparser_type::max_size;
      auto nPages = size / pageSize + (size % pageSize ? 1 : 0);
      if (!preCheck(ref.payload, dh->subSpecification)) {
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

      // check if the last block contains a valid RDH, otherwise data is corrupted or 8kb assumption is wrong
      if (!o2::raw::RDHUtils::checkRDH(ref.payload, false) || (nPages > 1 && (o2::raw::RDHUtils::getMemorySize(ref.payload) != pageSize || !o2::raw::RDHUtils::checkRDH(ref.payload + (nPages - 1) * pageSize, false)))) {
        forwardInternal(std::forward<Predicate>(pred), std::forward<Inserter>(inserter), ref.payload, size, dh);
        retVal = 1;
        continue;
      }

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
    return retVal;
  }

  template <typename Predicate, typename Inserter>
  int forward(Predicate pred, Inserter inserter)
  {
    return forward(std::forward<Predicate>(pred), std::forward<Inserter>(inserter), [](...) { return true; });
  }

  template <typename Predicate, typename Inserter, typename Precheck>
  int forward(Predicate pred, Inserter inserter, Precheck preCheck)
  {
    for (auto const& ref : mInput) {
      auto size = DataRefUtils::getPayloadSize(ref);
      if (size == 0) {
        continue;
      }
      auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (!preCheck(ref.payload, dh->subSpecification)) {
        continue;
      }
      forwardInternal(std::forward<Predicate>(pred), std::forward<Inserter>(inserter), ref.payload, size, dh);
    }
    return 0;
  }

 private:
  InputRecordWalker mInput;

  template <typename Predicate, typename Inserter>
  void forwardInternal(Predicate pred, Inserter inserter, const char* data, size_t size, const o2::header::DataHeader* dh)
  {
    o2::framework::RawParser parser(data, size);
    const char* ptr = nullptr;
    int count = 0;
    for (auto it = parser.begin(); it != parser.end(); it++) {
      const char* current = reinterpret_cast<const char*>(it.raw());
      if (ptr == nullptr) {
        ptr = current;
      } else if (pred(ptr, current) == false) {
        if (count) {
          inserter(ptr, count, dh->subSpecification);
        }
        count = 0;
        ptr = current;
      }
      count++;
      if (it.sizeTotal() != rawparser_type::max_size) {
        inserter(ptr, count, dh->subSpecification);
        count = 0;
        ptr = nullptr;
      }
    }
    if (count) {
      inserter(ptr, count, dh->subSpecification);
    }
  }
};

} // namespace o2::framework

#endif //FRAMEWORK_UTILS_DPLRAWPAGESEQUENCER_H
