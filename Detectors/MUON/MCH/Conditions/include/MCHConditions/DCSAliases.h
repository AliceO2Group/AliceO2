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

#ifndef O2_MCH_CONDITIONS_DCS_ALIASES_H
#define O2_MCH_CONDITIONS_DCS_ALIASES_H

#include "MCHConditions/Cathode.h"
#include "MCHConditions/Chamber.h"
#include "MCHConditions/DetectionElement.h"
#include "MCHConditions/MeasurementType.h"
#include "MCHConditions/Number.h"
#include "MCHConditions/Plane.h"
#include "MCHConditions/Side.h"
#include <string>
#include <vector>

namespace o2::mch::dcs
{

/* aliases gets a list of MCH DCS aliases for the given measurement type(s).
 * @param types a vector of the measurement types for which the aliases should
 * be returned.
 * @returns a list of MCH DCS alias names.
 */
std::vector<std::string> aliases(std::vector<MeasurementType> types = {
                                   MeasurementType::HV_V,
                                   MeasurementType::HV_I,
                                   MeasurementType::LV_V_FEE_ANALOG,
                                   MeasurementType::LV_V_FEE_DIGITAL,
                                   MeasurementType::LV_V_SOLAR});

/* check if alias is a valid alias */
bool isValid(std::string_view dcsAlias);

template <typename T>
std::ostream& operator<<(std::ostream& os, T& object)
{
  os << name(object);
  return os;
}

} // namespace o2::mch::dcs

#endif
