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

#ifndef O2_MCH_ROFFILTERING_MULTIPLICITY_FILTER_H_
#define O2_MCH_ROFFILTERING_MULTIPLICITY_FILTER_H_

#include <functional>
#include "DataFormatsMCH/ROFRecord.h"
#include "MCHROFFiltering/ROFFilter.h"

namespace o2::mch
{
/** Returns a basic ROFRecord filter that selects ROFs with a minimum number
 * of entries.
 *
 * The returned filter is a function that takes a ROFRecord and returns
 * a boolean.
 *
 * @param minMultiplicity
 *
 */

inline ROFFilter
  createMultiplicityFilter(int minMultiplicity)
{
  return [minMultiplicity](const ROFRecord& rof) {
    return rof.getNEntries() >= minMultiplicity;
  };
}

} // namespace o2::mch

#endif
