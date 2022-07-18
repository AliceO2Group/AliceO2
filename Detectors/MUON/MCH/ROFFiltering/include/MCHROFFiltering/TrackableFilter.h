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

#ifndef O2_MCH_ROFFILTERING_TRACKABLE_FILTER_H_
#define O2_MCH_ROFFILTERING_TRACKABLE_FILTER_H_

#include <functional>
#include "DataFormatsMCH/ROFRecord.h"
#include <array>
#include <gsl/span>
#include "MCHBase/Trackable.h"
#include "MCHROFFiltering/ROFFilter.h"

namespace o2::mch
{
/** Returns a ROFRecord filter that selects ROFs that are trackable.
 *
 * The returned filter is a function that takes a ROFRecord and returns
 * a boolean.
 *
 * @param items : the items "pointed to" by the ROFRecords (digits, ...)
 *
 * @param requestStation : @ref isTrackable
 * @param moreCandidates : @ref isTrackable
 *
 * @tparam : the type of the items pointed to by the ROFRecords
 */

template <typename T>
ROFFilter
  createTrackableFilter(gsl::span<const T> items,
                        std::array<bool, 5> requestStation = {true, true, true, true, true},
                        bool moreCandidates = false)
{
  return [items, requestStation, moreCandidates](const ROFRecord& rof) {
    std::array<int, 10> nofItemsPerChamber = perChamber(items.subspan(rof.getFirstIdx(), rof.getNEntries()));
    return isTrackable(nofItemsPerChamber, requestStation, moreCandidates);
  };
}

} // namespace o2::mch

#endif
