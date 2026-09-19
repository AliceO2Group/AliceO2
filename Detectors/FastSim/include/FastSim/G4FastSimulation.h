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
///                             G4.fastSimEnvelope=AFaM"
///
/// `G4.fastSimEnvelope` names the VOLUME the model stands in for. The regions
/// Geant4 needs in order to consult the model are derived from it by walking its
/// subtree and collecting the media (FastSimRegions.h) -- a region in O2 can only
/// be "every volume of a given material", so covering a module means naming all
/// of its materials, and that list should not be maintained by hand.
///
/// `G4.fastSimRegions` overrides the walk with an explicit space-separated list
/// of media, for when a model should see less than a whole subtree.

#include "TG4RunConfiguration.h"
#include "TG4VUserFastSimulation.h"
#include "TG4VUserPostDetConstruction.h"

#include <string>
#include <vector>

namespace o2::fastsim
{

/// Creates and registers the models named in `G4.fastSimModels`.
class G4FastSimulation : public TG4VUserFastSimulation
{
 public:
  G4FastSimulation(std::vector<std::string> models, const std::string& envelope,
                   double minEnergyGeV);
  void Construct() override;

 private:
  std::vector<std::string> mModels;
  std::string mEnvelope;
  double mMinEnergy = 1.;
};

/// The one hook O2 was missing. Returns nullptr when no model is configured,
/// which is exactly the behaviour before this file existed.
class G4RunConfiguration : public TG4RunConfiguration
{
 public:
  using TG4RunConfiguration::TG4RunConfiguration;
  TG4VUserFastSimulation* CreateUserFastSimulation() override;
  TG4VUserPostDetConstruction* CreateUserPostDetConstruction() override;
};

} // namespace o2::fastsim

#endif // O2_FASTSIM_G4_FAST_SIMULATION_H_
