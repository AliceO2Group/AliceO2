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

#include "MCHConditions/DCSAliases.h"

#include "HVAliases.h"
#include "LVAliases.h"
#include "MCHConstants/DetectionElements.h"
#include <array>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <set>

namespace
{

bool in(o2::mch::dcs::MeasurementType type,
        const std::vector<o2::mch::dcs::MeasurementType>& types)
{
  return std::find(begin(types), end(types), type) != types.end();
}

} // namespace

namespace o2::mch::dcs
{

std::vector<std::string> aliases(std::vector<MeasurementType> types)
{
  std::vector<std::string> aliasesVector;

  if (in(MeasurementType::HV_V, types)) {
    aliasesVector.insert(end(aliasesVector),
                         begin(expectedHVAliasesVoltages),
                         end(expectedHVAliasesVoltages));
  }
  if (in(MeasurementType::HV_I, types)) {
    aliasesVector.insert(end(aliasesVector),
                         begin(expectedHVAliasesCurrents),
                         end(expectedHVAliasesCurrents));
  }
  if (in(MeasurementType::LV_V_FEE_ANALOG, types)) {
    aliasesVector.insert(end(aliasesVector),
                         begin(expectedLVAliasesFeeAnalog),
                         end(expectedLVAliasesFeeAnalog));
  }
  if (in(MeasurementType::LV_V_FEE_DIGITAL, types)) {
    aliasesVector.insert(end(aliasesVector),
                         begin(expectedLVAliasesFeeDigital),
                         end(expectedLVAliasesFeeDigital));
  }
  if (in(MeasurementType::LV_V_SOLAR, types)) {
    aliasesVector.insert(end(aliasesVector),
                         begin(expectedLVAliasesSolar),
                         end(expectedLVAliasesSolar));
  }
  return aliasesVector;
}

std::vector<std::string> allAliases()
{
  std::vector<std::string> validAliases;
  validAliases.insert(validAliases.end(), expectedHVAliasesVoltages.begin(),
                      expectedHVAliasesVoltages.end());
  validAliases.insert(validAliases.end(), expectedHVAliasesCurrents.begin(),
                      expectedHVAliasesCurrents.end());
  validAliases.insert(validAliases.end(), expectedLVAliasesFeeAnalog.begin(),
                      expectedLVAliasesFeeAnalog.end());
  validAliases.insert(validAliases.end(), expectedLVAliasesFeeDigital.begin(),
                      expectedLVAliasesFeeDigital.end());
  validAliases.insert(validAliases.end(), expectedLVAliasesSolar.begin(),
                      expectedLVAliasesSolar.end());
  return validAliases;
}

bool isValid(std::string_view dcsAlias)
{
  static std::vector<std::string> validAliases = allAliases();
  return std::find(validAliases.begin(), validAliases.end(), dcsAlias) != validAliases.end();
}

} // namespace o2::mch::dcs
