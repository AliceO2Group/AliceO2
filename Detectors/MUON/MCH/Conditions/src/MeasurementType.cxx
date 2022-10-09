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

#include "MCHConditions/MeasurementType.h"
#include <array>
#include <fmt/core.h>

namespace o2::mch::dcs
{
std::string name(o2::mch::dcs::MeasurementType m)
{
  switch (m) {
    case o2::mch::dcs::MeasurementType::HV_V:
      return "vMon";
    case o2::mch::dcs::MeasurementType::HV_I:
      return "iMon";
    case o2::mch::dcs::MeasurementType::LV_V_FEE_ANALOG:
      return "an";
    case o2::mch::dcs::MeasurementType::LV_V_FEE_DIGITAL:
      return "di";
    case o2::mch::dcs::MeasurementType::LV_V_SOLAR:
      return "Sol";
  }
  return "INVALID";
}
MeasurementType aliasToMeasurementType(std::string_view alias)
{
  std::array<MeasurementType, 5> measurements = {
    MeasurementType::HV_V,
    MeasurementType::HV_I,
    MeasurementType::LV_V_FEE_ANALOG,
    MeasurementType::LV_V_FEE_DIGITAL,
    MeasurementType::LV_V_SOLAR};

  for (const auto m : measurements) {
    const auto mname = name(m);
    if (alias.find(mname) != std::string::npos) {
      return m;
    }
  }
  throw std::invalid_argument(fmt::format("Cannot infer the measurementType for alias={}", alias));
}

} // namespace o2::mch::dcs
