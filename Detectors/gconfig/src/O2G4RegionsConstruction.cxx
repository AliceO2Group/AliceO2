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

#include <G4LogicalVolume.hh>
#include <G4Region.hh>

#include <TG4GeometryServices.h>

#include <FairLogger.h>

#include "SimSetup/O2G4RegionsConstruction.h"
#include "DetectorsBase/RegionsManager.h"
#include "FastSimBase/DirectTransport.h"
#include "FastSimBase/DirectTransportBuilder.h"

using namespace o2;

void O2G4RegionsConstruction::Construct()
{
  auto regionVolumesMap = base::RegionsManager::Instance().getRegionVolumesMap();
  auto& directTransportBuilder = base::DirectTransportBuilder::Instance();
  for(auto& rVols : regionVolumesMap) {
    if(rVols.second.size() == 0) {
      continue;
    }
    auto region = new G4Region(rVols.first);
    LOG(INFO) << "Constructed new GEANT4 region " << rVols.first;
    // Now loop over volumes
    for(auto& vol : rVols.second) {
      auto g4LV = TG4GeometryServices::Instance()->FindLogicalVolume(vol.c_str());
      if(!g4LV) {
        LOG(FATAL) << "The GEANT4 corresponding volume " << vol << " cannot be found.";
      }
      region->AddRootLogicalVolume(g4LV);
      LOG(INFO) << "Added logical volume " << vol << " to GEANT4 region " << rVols.first;
    }
    // Build the direct transport
    LOG(INFO) << "Build direct transport for GEANT4 region " << rVols.first;
    directTransportBuilder.build(region);
  }
}
