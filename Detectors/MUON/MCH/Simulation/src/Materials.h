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
/// \brief  Implementation of the MCH materials definitions
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   22 mars 2018

#ifndef O2_MCH_SIMULATION_MATERIALS_H
#define O2_MCH_SIMULATION_MATERIALS_H

#include <TGeoMedium.h>

namespace o2
{

namespace mch
{

enum Medium {
  Gas,
  Carbon,
  HoneyNomex,
  BulkNomex,
  Noryl,
  Copper,
  FR4,
  Rohacell,
  Glue,
  Plastic,
  Epoxy,
  Inox,
  St1Rohacell,
  Aluminium,
  Air
};

// Return a pointer to the mch medium number imed.
// Throws an exception if imed is not within Medium enum
// and / or medium has not been defined yet.
TGeoMedium* assertMedium(int imed);

void createMaterials();

} // namespace mch
} // namespace o2

#endif // O2_MCH_SIMULATION_MATERIALS_H
