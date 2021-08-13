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

/// @file  MatchingType.h
/// \brief Defintions for the inter-detector matching type
/// \author ruben.shahoyan@cern.ch

#ifndef O2_MATCHING_TYPE
#define O2_MATCHING_TYPE

namespace o2
{
namespace globaltracking
{
enum class MatchingType {
  Standard, // standard matching, i.e. no extended workflow was applied
  Full,     // device is in the full matching mode
  Strict,   // device is in the strict matching mode
  NModes
};

static constexpr uint32_t getSubSpec(MatchingType t)
{
  return t == MatchingType::Strict ? 1 : 0; // Only strict matching inputs and outputs need special SubSpec
  if (t == MatchingType::Standard) {
    return 0;
  }
}

} // namespace globaltracking
} // namespace o2

#endif
