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

namespace o2
{
namespace mid
{
ChamberHV createDefaultChamberHV()
{
  /// Creates the default chamber voltages
  ChamberHV hv;
  for (int ide = 0; ide < detparams::NDetectionElements; ++ide) {
    hv.setHV(ide, 9800.);
  }

  return std::move(hv);
}

} // namespace mid
} // namespace o2
