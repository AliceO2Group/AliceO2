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

#ifndef O2_MUON_COMMON_ALIASFIXER_H
#define O2_MUON_COMMON_ALIASFIXER_H

#include <string>

namespace o2::muon
{
/** For some reason linked to ADAPOS (or underlying ORACLE ?)
  * datapoints we get from there cannot contain the dot character,
  * so we replace it by an underscore.
  */
std::string replaceDotByUnderscore(const std::string& alias);
}; // namespace o2::muon

#endif
