// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Materials.h
/// \brief  Implementation of the MID materials definitions
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   20 june 2018

#ifndef O2_MID_SIMULATION_MATERIALS_H
#define O2_MID_SIMULATION_MATERIALS_H

#include <TGeoMedium.h>

namespace o2
{

namespace mid
{

enum Medium {
  Gas,
  Bakelite,
  Inox,
  Aluminium,
  Copper,
  Mylar,
  Styrofoam,
  Nomex
};

// Return a pointer to the mid medium number imed.
// Throws an exception if imed is not within Medium enum
// and / or medium has not been defined yet.
TGeoMedium* assertMedium(int imed);

void createMaterials();

} // namespace mid
} // namespace o2

#endif // O2_MID_SIMULATION_MATERIALS_H
