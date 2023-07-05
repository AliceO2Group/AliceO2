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

/// @file  FDDDCSDataProcessor.cxx
/// @brief Task for processing FDD DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "FDDDCSMonitoring/FDDDCSDataProcessor.h"

#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include <string>
#include <vector>

std::vector<o2::dcs::DataPointIdentifier> o2::fdd::FDDDCSDataProcessor::getHardCodedDPIDs()
{
  std::vector<o2::dcs::DataPointIdentifier> vect;
  std::vector<std::string> aliasesHV = {"FDD/SIDE_A/HV_A9/[I,V]MON",
                                        "FDD/SIDE_C/HV_C[9,32]/[I,V]MON",
                                        "FDD/SIDE_C/LAYER0/PMT_0_[0..3]/[I,V]MON",
                                        "FDD/SIDE_C/LAYER1/PMT_1_[0..3]/[I,V]MON",
                                        "FDD/SIDE_A/LAYER2/PMT_2_[0..3]/[I,V]MON",
                                        "FDD/SIDE_A/LAYER3/PMT_3_[0..3]/[I,V]MON"};
  std::vector<std::string> aliasesADC = {"FDD/PM/SIDE_A/PMT_A_9/ADC[0,1]_BASELINE",
                                         "FDD/PM/SIDE_C/PMT_C_[9,32]/ADC[0,1]_BASELINE",
                                         "FDD/PM/SIDE_C/LAYER0/PMT_0_[0..3]/ADC[0,1]_BASELINE",
                                         "FDD/PM/SIDE_C/LAYER1/PMT_1_[0..3]/ADC[0,1]_BASELINE",
                                         "FDD/PM/SIDE_A/LAYER2/PMT_2_[0..3]/ADC[0,1]_BASELINE",
                                         "FDD/PM/SIDE_A/LAYER3/PMT_3_[0..3]/ADC[0,1]_BASELINE"};
  std::vector<std::string> aliasesRates = {"FDD/Trigger1_Central/CNT_RATE",
                                           "FDD/Trigger2_SemiCentral/CNT_RATE",
                                           "FDD/Trigger3_Vertex/CNT_RATE",
                                           "FDD/Trigger4_OrC/CNT_RATE",
                                           "FDD/Trigger5_OrA/CNT_RATE",
                                           "FDD/Background/[0..9]/CNT_RATE",
                                           "FDD/Background/[A,B,C,D,E,F,G,H]/CNT_RATE"};
  std::vector<std::string> expAliasesHV = o2::dcs::expandAliases(aliasesHV);
  std::vector<std::string> expAliasesADC = o2::dcs::expandAliases(aliasesADC);
  std::vector<std::string> expAliasesRates = o2::dcs::expandAliases(aliasesRates);
  for (const auto& i : expAliasesHV) {
    vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
  }
  for (const auto& i : expAliasesADC) {
    vect.emplace_back(i, o2::dcs::DPVAL_UINT);
  }
  for (const auto& i : expAliasesRates) {
    vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
  }
  return vect;
}
