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

/// @file  FT0DCSDataProcessor.h
/// @brief Task for processing FT0 DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "FT0DCSMonitoring/FT0DCSDataProcessor.h"

#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include <string>
#include <vector>

std::vector<o2::dcs::DataPointIdentifier> o2::ft0::FT0DCSDataProcessor::getHardCodedDPIDs()
{
  std::vector<o2::dcs::DataPointIdentifier> vect;
  std::vector<std::string> aliasesHV = {"FT0/HV/FT0_A/MCP_A[1..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_B[1..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_C[1..2]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_C[4..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_D[1..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_E[1..5]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_A[2..5]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_B[1..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_C[1..2]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_C[5..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_D[1..2]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_D[5..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_E[1..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_F[2..5]/actual/iMon",
                                        "FT0/HV/MCP_LC/actual/iMon"};
  std::string aliasesADC = "FT0/PM/channel[000..211]/actual/ADC[0..1]_BASELINE";
  std::vector<std::string> aliasesRates = {"FT0/Trigger1_Central/CNT_RATE",
                                           "FT0/Trigger2_SemiCentral/CNT_RATE",
                                           "FT0/Trigger3_Vertex/CNT_RATE",
                                           "FT0/Trigger4_OrC/CNT_RATE",
                                           "FT0/Trigger5_OrA/CNT_RATE",
                                           "FT0/Background/[0..9]/CNT_RATE",
                                           "FT0/Background/[A,B,C,D,E,F,G,H]/CNT_RATE",
                                           "FT0/SecondaryCounter/CEplusSC/CNT_RATE"};
  std::vector<std::string> expAliasesHV = o2::dcs::expandAliases(aliasesHV);
  std::vector<std::string> expAliasesADC = o2::dcs::expandAlias(aliasesADC);
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
