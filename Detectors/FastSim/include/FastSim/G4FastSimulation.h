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

#ifndef O2_FASTSIM_G4_FAST_SIMULATION_H_
#define O2_FASTSIM_G4_FAST_SIMULATION_H_

/// Wiring of the fast simulation models into the Geant4 engine.
///
/// The feature is OFF unless `G4.fastSimModels` names a model.
///
///   o2-sim -n 10 -g pythia8pp -e TGeant4 -m PIPE ABSO
///          --configKeyValues "G4.fastSimModels=toyAbsorber;
///                             G4.fastSimRegions=ABSO_AIR_ENVELOPE"
///
/// Regions are selected by TRACKING MEDIUM name (wildcards allowed), which
/// Geant4-VMC maps to the medium's MATERIAL; every volume of that material joins
/// the region. That is why the absorber mother volume AFaM carries a material of
/// its own (see Detectors/Passive/src/Absorber.cxx). Selection by volume is not
/// available: the VMC special cuts already root every logical volume in a
/// per-material region, and Geant4 allows a volume in only one region.

#include "TG4RunConfiguration.h"
#include "TG4VUserFastSimulation.h"

#include <string>
#include <vector>

namespace o2::fastsim
{

/// Creates and registers the models named in `G4.fastSimModels`.
class G4FastSimulation : public TG4VUserFastSimulation
{
 public:
  G4FastSimulation(std::vector<std::string> models, const std::string& regions,
                   double minEnergyGeV);
  void Construct() override;

 private:
  std::vector<std::string> mModels;
  double mMinEnergy = 1.;
};

/// The one hook O2 was missing. Returns nullptr when no model is configured,
/// which is exactly the behaviour before this file existed.
class G4RunConfiguration : public TG4RunConfiguration
{
 public:
  using TG4RunConfiguration::TG4RunConfiguration;
  TG4VUserFastSimulation* CreateUserFastSimulation() override;
};

} // namespace o2::fastsim

#endif // O2_FASTSIM_G4_FAST_SIMULATION_H_
