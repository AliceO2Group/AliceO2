// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MID_CONDITIONS_DCS_NAMER_H
#define O2_MID_CONDITIONS_DCS_NAMER_H

#include <vector>
#include <string>
#include <optional>
#include <cstdint>

namespace o2::mid::dcs
{
// The two types of MID DCS measurements that are of interest for reconstruction
enum class MeasurementType {
  HV_V, // HV voltage
  HV_I  // HV current
};

// Side describes on which side (inside or outside) a RPC is.
enum class Side {
  Inside,
  Outside
};

// ID is used to reference a RPC in MID DCS.
struct ID {
  int number;
  Side side;
  int chamberId; // 11,12,21,22
};

// detElemId2DCS converts a detection element id into a dcs-id.
// @returns an ID if deId is a valid the MID detID, std::nullopt otherwise.
std::optional<ID> detElemId2DCS(int deId);

// aliases gets a list of MID DCS aliases for the given measurement type(s).
// @param types a vector of the measurement types for which the aliases should
// be returned.
// @returns a list of MID DCS alias names.
std::vector<std::string> aliases(std::vector<MeasurementType> types = {
                                   MeasurementType::HV_V,
                                   MeasurementType::HV_I});

} // namespace o2::mid::dcs

#endif
