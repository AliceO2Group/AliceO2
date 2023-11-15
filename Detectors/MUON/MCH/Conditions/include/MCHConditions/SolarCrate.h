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

#ifndef O2_MCH_CONDITIONS_SOLAR_CRATE_H
#define O2_MCH_CONDITIONS_SOLAR_CRATE_H

#include <string_view>

namespace o2::mch::dcs
{
/** extract the solar crate from the alias.
 * alias must be valid (and of solar type) otherwise the methods throws an exception. */
int aliasToSolarCrate(std::string_view alias);
} // namespace o2::mch::dcs

#endif
