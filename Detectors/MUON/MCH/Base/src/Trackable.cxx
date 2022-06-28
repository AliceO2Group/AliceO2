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

#include "MCHBase/Trackable.h"
#include "DataFormatsMCH/Digit.h"

namespace o2::mch
{

bool isTrackable(std::array<int, 10> itemsPerChamber,
                 std::array<bool, 5> requestStation,
                 bool moreCandidates)
{
  // first check that the required stations are actually hit
  for (auto i = 0; i < 5; i++) {
    int inStation = itemsPerChamber[i * 2] + itemsPerChamber[i * 2 + 1];
    if (requestStation[i] && inStation == 0) {
      return false;
    }
  }
  // then check that we have the right number of hit chambers in St45
  int nChHitInSt4 = (itemsPerChamber[6] > 0 ? 1 : 0) + (itemsPerChamber[7] > 0 ? 1 : 0);
  int nChHitInSt5 = (itemsPerChamber[8] > 0 ? 1 : 0) + (itemsPerChamber[9] > 0 ? 1 : 0);

  if (moreCandidates) {
    return nChHitInSt4 + nChHitInSt5 >= 2;
  } else {
    return nChHitInSt4 == 2 || nChHitInSt5 == 2;
  }
  return true;
}

/** Specialization of perChamber for integers (representing
 * detection element ids.
 */
template <>
std::array<int, 10> perChamber(gsl::span<const int> deids)
{
  std::array<int, 10> nitems{};
  for (const auto& d : deids) {
    nitems[d / 100 - 1]++;
  }
  return nitems;
}

/** Specialization of perChamber for Digits */
template <>
std::array<int, 10> perChamber(gsl::span<const Digit> digits)
{
  std::array<int, 10> nofDigits{};
  for (const auto& digit : digits) {
    nofDigits[digit.getDetID() / 100 - 1]++;
  }
  return nofDigits;
}

} // namespace o2::mch
