// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_CONDITIONS_DCS_NAMER_H
#define O2_MCH_CONDITIONS_DCS_NAMER_H

#include <vector>
#include <string>
#include <optional>
#include <cstdint>

namespace o2::mch::dcs
{
// The list of MCH DCS measurements that are of interest for reconstruction
enum class MeasurementType {
  HV_V,             // HV voltage
  HV_I,             // HV current
  LV_V_FEE_ANALOG,  // FEE (dualsampa) analog voltage
  LV_V_FEE_DIGITAL, // FEE (dualsampa) digital voltage
  LV_V_SOLAR        // Solar crate voltage
};

// Side describes on which side (inside or outside) a detection element
// (slat or quadrant) is.
// Note that MCH DCS uses the very old left-right convention instead of the
// agreed-upon inside-outside.
enum class Side {
  Left,
  Right
};

// ID is used to reference a particular device in MCH DCS,
// like a detection element or solar crate for instance.
struct ID {
  int number;
  Side side;
  int chamberId; // 0..9
};

// detElemId2DCS converts a detection element id into a dcs-id.
// @returns an ID if deId is a valid the MCH detID, std::nullopt otherwise.
std::optional<ID> detElemId2DCS(int deId);

// aliases gets a list of MCH DCS aliases for the given measurement type(s).
// @param types a vector of the measurement types for which the aliases should
// be returned.
// @returns a list of MCH DCS alias names.
std::vector<std::string> aliases(std::vector<MeasurementType> types = {
                                   MeasurementType::HV_V,
                                   MeasurementType::HV_I,
                                   MeasurementType::LV_V_FEE_ANALOG,
                                   MeasurementType::LV_V_FEE_DIGITAL,
                                   MeasurementType::LV_V_SOLAR});

} // namespace o2::mch::dcs

#endif
