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

#ifndef O2_MCH_CONDITIONS_MEASUREMENT_TYPE_H
#define O2_MCH_CONDITIONS_MEASUREMENT_TYPE_H

#include <string>

namespace o2::mch::dcs
{
/** The list of MCH DCS measurements that are of interest for reconstruction.*/
enum class MeasurementType {
  HV_V,             // HV voltage
  HV_I,             // HV current
  LV_V_FEE_ANALOG,  // FEE (dualsampa) analog voltage
  LV_V_FEE_DIGITAL, // FEE (dualsampa) digital voltage
  LV_V_SOLAR        // Solar crate voltage
};

/** name of MeasurementType (an,di ...) */
std::string name(MeasurementType m);

/** extract the measurement type information from the alias
 * alias must be valid otherwise the method throws an exception. */
MeasurementType aliasToMeasurementType(std::string_view alias);

} // namespace o2::mch::dcs

#endif
