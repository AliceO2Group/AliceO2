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

#ifndef O2_MCH_GLOBAL_MAPPING_QUADRANT_H
#define O2_MCH_GLOBAL_MAPPING_QUADRANT_H

#include "MCHConditions/Cathode.h"
#include <set>
#include <string>

namespace o2::mch::dcs::quadrant
{
Cathode lvAliasToCathode(std::string_view alias);
std::set<int> solarAliasToDsIndices(std::string_view alias);
} // namespace o2::mch::dcs::quadrant

#endif
