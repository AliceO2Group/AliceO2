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

/// \file FEEConfig.cxx
/// \brief Frontend electronics configuration values
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <string_view>
#include "Framework/Logger.h"
#include "CommonUtils/StringUtils.h"
#include "TPCBase/FEEConfig.h"

using namespace o2::utils;
using namespace o2::tpc;

bool CRUConfig::setValues(std::string_view cruData)
{
  const auto cruDataV = Str::tokenize(cruData.data(), ',');
  if (cruDataV.size() != CRUConfig::NConfigValues) {
    LOGP(warning, "Wrong number of CRU config values {}/{} in line {}", cruDataV.size(), NConfigValues, cruData);
    return false;
  }

  linkOn = bool(std::stoi(cruDataV[0]));
  cmcEnabled = bool(std::stoi(cruDataV[1]));
  itfEnabled = bool(std::stoi(cruDataV[2]));
  zsEnabled = bool(std::stoi(cruDataV[3]));
  zsOffset = std::stof(cruDataV[4]);
  itCorr0 = std::stof(cruDataV[5]);

  return true;
}
