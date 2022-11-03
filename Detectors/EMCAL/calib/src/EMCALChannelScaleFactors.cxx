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

#include "EMCALCalib/CalibContainerErrors.h"
#include "EMCALCalib/EMCALChannelScaleFactors.h"

namespace o2
{
namespace emcal
{

void EMCALChannelScaleFactors::insertVal(unsigned int cellID, float E_min, float E_max, float scale)
{
  if (cellID >= NCells || cellID < 0) {
    throw CalibContainerIndexException(cellID);
  } else {
    ScaleFactors.at(cellID)[EnergyIntervals(E_min, E_max)] = scale;
  }
}

float EMCALChannelScaleFactors::getScaleVal(unsigned int cellID, float E) const
{
  if (cellID >= NCells || cellID < 0) {
    throw CalibContainerIndexException(cellID);
  } else {
    for (const auto& [energy, scale] : ScaleFactors[cellID]) {
      if (energy.isInInterval(E)) {
        return scale;
      }
    }
    throw InvalidEnergyIntervalException(E, cellID);
  }
}
} // namespace emcal
} // namespace o2