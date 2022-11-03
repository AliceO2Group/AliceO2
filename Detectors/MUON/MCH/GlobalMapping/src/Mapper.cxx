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

#include "MCHGlobalMapping/Mapper.h"

#include "HV.h"
#include "LV.h"
#include "MCHConditions/DCSAliases.h"
#include "MCHConditions/SolarCrate.h"
#include "MCHConstants/DetectionElements.h"
#include "MCHGlobalMapping/DsIndex.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/Mapper.h"
#include "Quadrant.h"
#include "Slat.h"
#include <limits>
#include <map>
#include <vector>

namespace o2::mch::dcs
{

std::set<int> solarAliasToDsIndices(std::string_view alias)
{
  const auto chamber = aliasToChamber(alias);
  if (dcs::isQuadrant(chamber)) {
    return dcs::quadrant::solarAliasToDsIndices(alias);
  } else {
    return dcs::slat::solarAliasToDsIndices(alias);
  }
}

std::set<int> aliasToDsIndices(std::string_view alias)
{
  auto m = aliasToMeasurementType(alias);
  switch (m) {
    case dcs::MeasurementType::HV_I:
    case dcs::MeasurementType::HV_V:
      return hvAliasToDsIndices(alias);
    case dcs::MeasurementType::LV_V_FEE_ANALOG:
    case dcs::MeasurementType::LV_V_FEE_DIGITAL:
      return lvAliasToDsIndices(alias);
    case dcs::MeasurementType::LV_V_SOLAR:
      return solarAliasToDsIndices(alias);
    default:
      return {};
  }
}
} // namespace o2::mch::dcs
