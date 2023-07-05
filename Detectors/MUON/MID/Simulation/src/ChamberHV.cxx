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

/// \file   MID/Simulation/src/ChamberHV.cxx
/// \brief  Implementation of the HV for MID RPCs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 April 2018

#include "MIDSimulation/ChamberHV.h"

#include "MIDConditions/DCSNamer.h"

namespace o2
{
namespace mid
{

std::vector<std::pair<uint64_t, double>> getValues(const std::unordered_map<o2::dcs::DataPointIdentifier, std::vector<o2::dcs::DataPointValue>>& dpMap, std::string alias)
{
  union Converter {
    uint64_t raw_data;
    double value;
  } converter;
  std::vector<std::pair<uint64_t, double>> values;
  for (auto& dp : dpMap) {
    if (alias == dp.first.get_alias()) {
      for (auto& dpVal : dp.second) {
        converter.raw_data = dpVal.payload_pt1;
        values.emplace_back(dpVal.get_epoch_time(), converter.value);
      }
    }
  }
  return values;
}

double getAverage(std::vector<std::pair<uint64_t, double>>& values)
{
  double num = 0., den = 0.;
  for (size_t ival = 1; ival < values.size(); ++ival) {
    double delta = values[ival].first - values[ival - 1].first;
    num += values[ival - 1].second * delta;
    den += delta;
  }
  return (den > 0.) ? num / den : num;
}

void ChamberHV::setHV(const std::unordered_map<o2::dcs::DataPointIdentifier, std::vector<o2::dcs::DataPointValue>>& dpMap)
{
  for (int deId = 0; deId < detparams::NDetectionElements; ++deId) {
    auto alias = detElemId2DCSAlias(deId, dcs::MeasurementType::HV_V);
    auto values = getValues(dpMap, alias);
    auto hv = getAverage(values);
    setHV(deId, hv);
  }
}

ChamberHV createDefaultChamberHV()
{
  ChamberHV hv;
  for (int ide = 0; ide < detparams::NDetectionElements; ++ide) {
    hv.setHV(ide, 9600.);
  }

  return hv;
}

} // namespace mid
} // namespace o2
