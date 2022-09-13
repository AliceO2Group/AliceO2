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

/// @file  FV0DCSDataProcessor.cxx
/// @brief Task for processing FV0 DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "FV0DCSMonitoring/FV0DCSDataProcessor.h"

#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include <string>
#include <vector>

std::vector<o2::dcs::DataPointIdentifier> o2::fv0::FV0DCSDataProcessor::getHardCodedDPIDs()
{
  std::vector<o2::dcs::DataPointIdentifier> vect;
  std::vector<std::string> aliasesHV = {"FV0/HV/S[A,B,C,D,E,F,G,H][1..4]/actual/iMon",
                                        "FV0/HV/S[A,B,C,D,E,F,G,H][51,52]/actual/iMon",
                                        "FV0/HV/SREF/actual/iMon"};
  std::vector<std::string> aliasesADC = {"FV0/PM/S[A,B,C,D,E,F,G,H][1..4]/actual/ADC[0,1]_BASELINE",
                                         "FV0/PM/S[A,B,C,D,E,F,G,H][51,52]/actual/ADC[0,1]_BASELINE",
                                         "FV0/PM/SREF/actual/ADC[0,1]_BASELINE"};
  std::vector<std::string> expAliasesHV = o2::dcs::expandAliases(aliasesHV);
  std::vector<std::string> expAliasesADC = o2::dcs::expandAliases(aliasesADC);
  for (const auto& i : expAliasesHV) {
    vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
  }
  for (const auto& i : expAliasesADC) {
    vect.emplace_back(i, o2::dcs::DPVAL_UINT);
  }
  return vect;
}
