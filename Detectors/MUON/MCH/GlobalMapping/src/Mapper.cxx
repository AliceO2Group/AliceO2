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

std::set<DsIndex> getDsIndices(const std::set<dcs::Cathode>& cathodes)
{
  std::set<DsIndex> dsIndices;
  for (const auto& cathode : cathodes) {
    auto deId = cathode.deId;
    if (!constants::isValidDetElemId(deId)) {
      continue;
    }
    bool bending = cathode.plane == dcs::Plane::Bending;
    bool checkPlane = cathode.plane != dcs::Plane::Both;
    if (checkPlane) {
      const o2::mch::mapping::Segmentation& seg = o2::mch::mapping::segmentation(deId);
      const auto& plane = bending ? seg.bending() : seg.nonBending();
      for (auto i = 0; i < plane.nofDualSampas(); i++) {
        auto index = o2::mch::getDsIndex({deId, plane.dualSampaId(i)});
        dsIndices.emplace(index);
      }
    } else {
      const o2::mch::mapping::Segmentation& seg = o2::mch::mapping::segmentation(deId);
      seg.forEachDualSampa([&dsIndices, deId](int dualSampaId) {
        auto index = o2::mch::getDsIndex({deId, dualSampaId});
        dsIndices.emplace(index);
      });
    }
  }
  return dsIndices;
}

std::set<DsIndex> getDsIndices(dcs::Chamber ch, dcs::Plane plane)
{
  std::set<dcs::Cathode> cathodes;
  for (auto deid : constants::deIdsForAllMCH) {
    if (deid / 100 - 1 == toInt(ch)) {
      Cathode cathode{deid, plane};
      cathodes.insert(cathode);
    }
  }
  return getDsIndices(cathodes);
}

std::set<DsIndex> getDsIndices(const std::set<int>& solarIds)
{
  std::set<o2::mch::DsIndex> dualSampas;
  for (const auto& solarId : solarIds) {
    auto dsDetIds = o2::mch::raw::getDualSampas<o2::mch::raw::ElectronicMapperGenerated>(solarId);
    for (const auto& dsDetId : dsDetIds) {
      auto index = o2::mch::getDsIndex(dsDetId);
      dualSampas.emplace(index);
    }
  }
  return dualSampas;
}

} // namespace o2::mch::dcs
