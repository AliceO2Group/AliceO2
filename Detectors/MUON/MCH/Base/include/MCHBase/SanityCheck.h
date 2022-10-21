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

#ifndef O2_MCH_BASE_SANITY_CHECK_H
#define O2_MCH_BASE_SANITY_CHECK_H

#include "DataFormatsMCH/ROFRecord.h"
#include "Framework/Logger.h"
#include <gsl/span>
#include <map>
#include <string>
#include <vector>

namespace o2::mch
{
struct Pad {
  int detId;
  int padId;

  bool operator<(const Pad& other) const
  {
    if (detId == other.detId) {
      return padId < other.padId;
    }
    return detId < other.detId;
  }
};

struct SanityError {
  uint32_t nofDuplicatedItems{0};
  uint32_t nofDuplicatedIndices{0};
  uint32_t nofMissingItems{0};
  uint32_t nofOutOfBounds{0};
};

bool isOK(const SanityError& error);
std::string asString(const SanityError& error);

template <typename T>
SanityError sanityCheck(gsl::span<const ROFRecord> rofs, gsl::span<const T> items)
{
  // check that :
  // - all items indices are used in rofs
  // - each item is referenced only once
  // - all indices used in rofs are inbound of items.size()

  SanityError error;

  std::vector<int> indices;
  indices.resize(items.size());

  for (const auto& r : rofs) {
    std::map<T, int> itemMap;
    for (auto i = r.getFirstIdx(); i <= r.getLastIdx(); i++) {
      if (i >= items.size()) {
        error.nofOutOfBounds++;
        continue;
      }
      indices[i]++;
      const auto& item = items[i];
      itemMap[item]++;
    }
    if (itemMap.size() != r.getNEntries()) {
      error.nofDuplicatedItems += (r.getNEntries() - itemMap.size());
    }
  }
  for (auto i = 0; i < indices.size(); i++) {
    if (indices[i] == 0) {
      error.nofMissingItems++;
    }
    if (indices[i] > 1) {
      error.nofDuplicatedIndices++;
    }
  }

  return error;
}
} // namespace o2::mch

#endif
