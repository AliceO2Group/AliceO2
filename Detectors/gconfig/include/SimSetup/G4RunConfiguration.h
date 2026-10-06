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

#ifndef O2_SIMSETUP_G4RUNCONFIGURATION_H_
#define O2_SIMSETUP_G4RUNCONFIGURATION_H_

#include "TG4RunConfiguration.h"

namespace o2::g4config
{

/// The Geant4 VMC run configuration of O2: adds the fast simulation and the local magnetic fields.
class G4RunConfiguration : public TG4RunConfiguration
{
 public:
  using TG4RunConfiguration::TG4RunConfiguration;
  TG4VUserFastSimulation* CreateUserFastSimulation() override;
  TG4VUserPostDetConstruction* CreateUserPostDetConstruction() override;
};

} // namespace o2::g4config

#endif // O2_SIMSETUP_G4RUNCONFIGURATION_H_
