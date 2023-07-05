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

/// \file ECSDataAdapter
/// \brief Header only, adapter methods without external dependencies
/// \author ruben.shahoyan@cern.ch, gvozden.neskovic@cern.ch

#ifndef ALICEO2_ECSADAPTERS_H_
#define ALICEO2_ECSADAPTERS_H_

// NOTE: please only stdlib includes here!
#include <string>
#include <string_view>
#include <array>

namespace o2
{
namespace parameters
{

namespace GRPECS
{

enum RunType : int {
  NONE,
  PHYSICS,
  TECHNICAL,
  PEDESTAL,
  PULSER,
  LASER,
  CALIBRATION_ITHR_TUNING,
  CALIBRATION_VCASN_TUNING,
  CALIBRATION_THR_SCAN,
  CALIBRATION_DIGITAL_SCAN,
  CALIBRATION_ANALOG_SCAN,
  CALIBRATION_FHR,
  CALIBRATION_ALPIDE_SCAN,
  CALIBRATION,
  COSMICS,
  SYNTHETIC,
  NOISE,
  CALIBRATION_PULSE_LENGTH,
  CALIBRATION_VRESETD,
  NRUNTYPES
};
static constexpr std::array<std::string_view, NRUNTYPES> RunTypeNames = {
  "NONE",
  "PHYSICS",
  "TECHNICAL",
  "PEDESTAL",
  "PULSER",
  "LASER",
  "CALIBRATION_ITHR_TUNING",
  "CALIBRATION_VCASN_TUNING",
  "CALIBRATION_THR_SCAN",
  "CALIBRATION_DIGITAL_SCAN",
  "CALIBRATION_ANALOG_SCAN",
  "CALIBRATION_FHR",
  "CALIBRATION_ALPIDE_SCAN",
  "CALIBRATION",
  "COSMICS",
  "SYNTHETIC",
  "NOISE",
  "CALIBRATION_PULSE_LENGTH",
  "CALIBRATION_VRESETD"};

//_______________________________________________
static RunType string2RunType(const std::string& rts)
{
  int rt = -1;
  for (int i = 0; i < int(RunType::NRUNTYPES); i++) {
    if (rts == RunTypeNames[i]) {
      rt = i;
      break;
    }
  }
  return RunType(rt);
}

//_______________________________________________
static std::string getRawDataPersistencyMode(const std::string& runType, bool imposeRaw = false)
{
  if (imposeRaw) {
    return "raw";
  }
  std::string ret = "other";
  auto rt = string2RunType(runType);
  switch (rt) {
    case RunType::PHYSICS:
    case RunType::COSMICS:
      ret = "raw";
      break;
    case RunType::PEDESTAL:
    case RunType::PULSER:
    case RunType::LASER:
    case RunType::CALIBRATION_ITHR_TUNING:
    case RunType::CALIBRATION_VCASN_TUNING:
    case RunType::CALIBRATION_THR_SCAN:
    case RunType::CALIBRATION_DIGITAL_SCAN:
    case RunType::CALIBRATION_ANALOG_SCAN:
    case RunType::CALIBRATION_FHR:
    case RunType::CALIBRATION_ALPIDE_SCAN:
    case RunType::CALIBRATION:
    case RunType::CALIBRATION_PULSE_LENGTH:
    case RunType::CALIBRATION_VRESETD:
      ret = "calib";
    default:
      break;
  }
  return ret;
}

} // namespace GRPECS
} // namespace parameters
} // namespace o2

#endif
