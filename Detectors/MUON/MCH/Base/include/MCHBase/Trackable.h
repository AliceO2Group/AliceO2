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

#ifndef O2_MCH_TRACKABLE_H
#define O2_MCH_TRACKABLE_H

#include <array>
#include <functional>
#include <gsl/span>

namespace o2::mch
{

/** Given a number of items (digits, (pre)clusters) per chamber,
 * decides if there's the minimum required information to
 * get any chance of producing tracks.
 *
 * @param itemsPerChamber integer array containing the number
 * of items per chamber (10 chambers, so 10 integers required)
 * @param requestStation boolean array to indicate which stations
 * are required for the tracking (by default all 5 are needed)
 * @param moreCandidates weaker tracking condition where only
 * one item in each of station 4 and 5 is requested (instead of
 * one item per chamber in station 4 and in station 5)
 * */
bool isTrackable(std::array<int, 10> itemsPerChamber,
                 std::array<bool, 5> requestStation = {true, true, true, true, true},
                 bool moreCandidates = false);

/** Return the number of items per chamber.
 *
 * @tparam T the type of items : implementation exists so far
 * only for mch::Digit (clusters and pre-clusters to come next)
 */
template <typename T>
std::array<int, 10> perChamber(gsl::span<const T> items);

/** Return the number of items per station (1 station==2 chambers). */
template <typename T>
std::array<int, 5> perStation(gsl::span<const T> items)
{
  std::array<int, 5> st{};
  std::array<int, 10> ch = perChamber(items);
  for (auto i = 0; i < st.size(); ++i) {
    st[i] = ch[2 * i] + ch[2 * i + 1];
  }
  return st;
}

} // namespace o2::mch

#endif
