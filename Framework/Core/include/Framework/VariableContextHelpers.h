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
#ifndef O2_FRAMEWORK_DATADESCRIPTORMATCHER_H_
#define O2_FRAMEWORK_DATADESCRIPTORMATCHER_H_

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/TimesliceSlot.h"
#include <variant>

namespace o2::framework
{

/// Helpers to fetch common variables
struct VariableContextHelpers {
  static inline uint32_t getFirstTFOrbit(data_matcher::VariableContext const& variables)
  {
    // tfCounter is always at register 14
    auto pval = std::get_if<uint32_t>(&variables.get(data_matcher::FIRSTTFORBIT_POS));
    if (pval == nullptr) {
      return -1;
    }
    return *pval;
  }

  static inline TimesliceId getTimeslice(data_matcher::VariableContext const& variables)
  {
    // timeslice is always at register 0
    auto pval = std::get_if<uint64_t>(&variables.get(data_matcher::STARTTIME_POS));
    if (pval == nullptr) {
      return TimesliceId{TimesliceId::INVALID};
    }
    return TimesliceId{*pval};
  }

  static inline uint32_t getRunNumber(data_matcher::VariableContext const& variables)
  {
    // firstTForbit is always at register 15
    auto pval = std::get_if<uint32_t>(&variables.get(data_matcher::RUNNUMBER_POS));
    if (pval == nullptr) {
      return -1;
    }
    return *pval;
  }

  static inline uint32_t getFirstTFCounter(data_matcher::VariableContext const& variables)
  {
    // tfCounter is always at register 14
    auto pval = std::get_if<uint32_t>(&variables.get(data_matcher::TFCOUNTER_POS));
    if (pval == nullptr) {
      return -1;
    }
    return *pval;
  }

  static inline uint64_t getCreationTime(data_matcher::VariableContext const& variables)
  {
    // creation time is always at register 14
    auto pval = std::get_if<uint64_t>(&variables.get(data_matcher::CREATIONTIME_POS));
    if (pval == nullptr) {
      return -1UL;
    }
    return *pval;
  }
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_DATADESCRIPTORMATCHER_H_
