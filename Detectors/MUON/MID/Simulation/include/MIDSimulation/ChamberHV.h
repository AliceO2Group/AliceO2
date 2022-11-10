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

/// \file   MIDSimulation/ChamberHV.h
/// \brief  HV values for MID RPCs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 April 2018
#ifndef O2_MID_CHAMBERHV_H
#define O2_MID_CHAMBERHV_H

#include <array>
#include <vector>
#include <unordered_map>

#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "MIDBase/DetectorParameters.h"

namespace o2
{
namespace mid
{
class ChamberHV
{
 public:
  /// \brief Gets HV for detection element
  /// \param deId Detection element ID
  /// \return Detection element HV
  double getHV(int deId) const { return mHV[deId]; }

  /// \brief Sets the HV for detection element
  /// \param deId Detection element ID
  /// \param hv High-Voltage value (V)
  void setHV(int deId, double hv) { mHV[deId] = hv; }

  /// \brief Sets the HV from the DCS data points
  /// \param dpMap Map with DCS data points
  void setHV(const std::unordered_map<o2::dcs::DataPointIdentifier, std::vector<o2::dcs::DataPointValue>>& dpMap);

 private:
  std::array<double, detparams::NDetectionElements> mHV; ///< High voltage values
};

/// \brief Creates the default chamber voltages
/// \return Default chamber HV values
ChamberHV createDefaultChamberHV();

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHAMBERHV_H */
