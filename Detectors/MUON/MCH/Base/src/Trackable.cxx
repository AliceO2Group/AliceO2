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
#include "MCHBase/PreCluster.h"

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
  // do not count isolated digits (at least 2 are required for a cluster)
  for (auto i = 0; i < 10; ++i) {
    if (nofDigits[i] == 1) {
      nofDigits[i] = 0;
    }
  }
  return nofDigits;
}

/** Specialization of perChamber for PreClusters */
template <>
std::array<int, 10> perChamber(gsl::span<const PreCluster> preclusters, gsl::span<const Digit> digits)
{
  std::array<int, 10> nofPreclusters{};
  for (const auto& precluster : preclusters) {
    // only consider preclusters made of at least 2 digits
    if (precluster.nDigits > 1) {
      nofPreclusters[digits[precluster.firstDigit].getDetID() / 100 - 1]++;
    }
  }
  return nofPreclusters;
}

} // namespace o2::mch
