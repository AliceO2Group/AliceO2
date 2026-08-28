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

#include "FastSim/G4FastSimulation.h"
#include "FastSim/FastSimRegions.h"
#include "FastSim/ToyAbsorberFastSim.h"
#include "SimConfig/G4Params.h"

#include <fairlogger/Logger.h>

#include <sstream>

namespace o2::fastsim
{

namespace
{
std::vector<std::string> split(const std::string& value, char sep)
{
  std::vector<std::string> out;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, sep)) {
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  return out;
}
} // namespace

//_____________________________________________________________________________
G4FastSimulation::G4FastSimulation(std::vector<std::string> models,
                                   const std::string& envelope, double minEnergyGeV)
  : TG4VUserFastSimulation(), mModels(std::move(models)), mEnvelope(envelope), mMinEnergy(minEnergyGeV)
{
  // Only the model itself can be declared here: this constructor runs before the
  // geometry exists, so the regions cannot be derived yet. They are set later by
  // FastSimRegionConstruction, in the window between geometry construction and
  // region creation.
  for (const auto& model : mModels) {
    SetModel(model);
    SetModelParticles(model, "all");
  }
}

//_____________________________________________________________________________
void G4FastSimulation::Construct()
{
  for (const auto& model : mModels) {
    if (model == "toyAbsorber") {
      LOG(info) << "fast simulation: registering model " << model << " on envelope '" << mEnvelope
                << "' above " << mMinEnergy << " GeV";
      Register(new ToyAbsorberFastSim(model, mEnvelope, mMinEnergy));
    } else {
      LOG(error) << "fast simulation: unknown model " << model << "; ignored";
    }
  }
}

//_____________________________________________________________________________
TG4VUserFastSimulation* G4RunConfiguration::CreateUserFastSimulation()
{
  const auto& params = o2::conf::G4Params::Instance();
  auto models = split(params.fastSimModels, ',');
  if (models.empty()) {
    return nullptr; // the default: no fast simulation, unchanged behaviour
  }
  LOG(info) << "fast simulation is ENABLED on envelope '" << params.fastSimEnvelope << "'";
  return new G4FastSimulation(std::move(models), params.fastSimEnvelope, params.fastSimMinEnergy);
}

//_____________________________________________________________________________
TG4VUserPostDetConstruction* G4RunConfiguration::CreateUserPostDetConstruction()
{
  const auto& params = o2::conf::G4Params::Instance();
  auto models = split(params.fastSimModels, ',');
  if (models.empty()) {
    return TG4RunConfiguration::CreateUserPostDetConstruction();
  }
  std::vector<FastSimRegionConstruction::ModelRegions> wanted;
  wanted.reserve(models.size());
  for (auto& model : models) {
    wanted.push_back({model, params.fastSimEnvelope, params.fastSimRegions});
  }
  return new FastSimRegionConstruction(std::move(wanted));
}

} // namespace o2::fastsim
