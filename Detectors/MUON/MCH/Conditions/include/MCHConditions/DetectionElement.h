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

#ifndef O2_MCH_CONDITIONS_DETECTION_ELEMENT_H
#define O2_MCH_CONDITIONS_DETECTION_ELEMENT_H

#include "MCHConditions/Chamber.h"
#include "MCHConditions/Side.h"
#include <optional>
#include <string>

namespace o2::mch::dcs
{
/** extract (if possible) the detection element id from the alias */
std::optional<int> aliasToDetElemId(std::string_view dcsAlias);

int detElemId(Chamber chamber, Side side, int number);

} // namespace o2::mch::dcs

#endif
