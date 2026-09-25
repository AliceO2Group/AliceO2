// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimSetup/G4RunConfiguration.h"
#include "FastSim/G4FastSimulation.h"

namespace o2::g4config
{

TG4VUserFastSimulation* G4RunConfiguration::CreateUserFastSimulation()
{
  return o2::fastsim::createFastSimulation();
}

TG4VUserPostDetConstruction* G4RunConfiguration::CreateUserPostDetConstruction()
{
  auto fastSimRegions = o2::fastsim::createFastSimRegionConstruction();
  return fastSimRegions ? fastSimRegions : TG4RunConfiguration::CreateUserPostDetConstruction();
}

} // namespace o2::g4config
