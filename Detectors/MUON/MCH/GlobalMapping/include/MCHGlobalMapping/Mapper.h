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

#ifndef O2_MCH_GLOBAL_MAPPING_MAPPER_H_
#define O2_MCH_GLOBAL_MAPPING_MAPPER_H_

#include "MCHConditions/Cathode.h"
#include "MCHConditions/Chamber.h"
#include "MCHConditions/Plane.h"
#include "MCHGlobalMapping/DsIndex.h"
#include <set>
#include <string>

namespace o2::mch::dcs
{
/** get the list of dual sampa indices corresponding to a given DCS Alias */
std::set<int> aliasToDsIndices(std::string_view alias);

/** get indices of all dual sampas of a set of cathodes (={deId,plane}).
 * returned set might be empty if input Cathodes are not valid ones. */
std::set<DsIndex> getDsIndices(const std::set<dcs::Cathode>& cathodes);

/** get indices of all dual sampas of given plane of a chamber */
std::set<DsIndex> getDsIndices(dcs::Chamber ch, dcs::Plane plane);

/** get indices of all dual sampas of a given set of solars */
std::set<DsIndex> getDsIndices(const std::set<int>& solarIds);

} // namespace o2::mch::dcs

#endif
