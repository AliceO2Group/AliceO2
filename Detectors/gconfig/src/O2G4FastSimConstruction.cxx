// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file G4RegionsConstruction.cxx
/// \brief Definition of the Regions to be created by GEANT4 class

#include <iostream>

#include <G4Region.hh>
#include <G4RegionStore.hh>

#include <FairLogger.h>

#include "FastSimBase/DirectTransportBuilder.h"
#include "FastSimBase/DirectTransport.h"

#include "SimSetup/O2G4FastSimConstruction.h"

using namespace o2;

O2G4FastSimConstruction::O2G4FastSimConstruction(const std::vector<std::string>& directTransportRegions)
  : TG4VUserFastSimulation()
{
  auto& directTransportBuilder = base::DirectTransportBuilder::Instance();
  for(auto& r : directTransportRegions) {
    if(std::find(mDirectTransportRegions.begin(), mDirectTransportRegions.end(), r) == mDirectTransportRegions.end()) {
      mDirectTransportRegions.push_back(r);
      auto model = directTransportBuilder.build(r);
      mDirectTransportModels.push_back(model);
      SetModel(model->GetName());
      SetModelParticles(model->GetName(), "all");
      SetModelRegions(model->GetName(), r);
    }
  }
}

void O2G4FastSimConstruction::Construct()
{
  std::cout << "Built TGeant4 fast sims" << std::endl;
  for(auto& model : mDirectTransportModels) {
    Register(model);
    LOG(INFO) << "Built direct transport for region " << model->GetName();
  }
}
