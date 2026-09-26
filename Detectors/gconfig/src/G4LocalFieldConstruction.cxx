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

#include "SimSetup/G4LocalFieldConstruction.h"
#include "SimConfig/G4Params.h"

#include "TG4GeometryManager.h"

#include <G4UIcommandTree.hh>
#include <G4UImanager.hh>
#include <TGeoManager.h>
#include <TVirtualMC.h>
#include <fairlogger/Logger.h>

#include <string>
#include <unordered_set>

namespace o2::g4config
{

namespace
{
// First volume in the subtree of vol whose medium has no magnetic field (ifield = 0)
const TGeoVolume* findZeroFieldVolume(const TGeoVolume* vol, std::unordered_set<const TGeoVolume*>& visited)
{
  if (!visited.insert(vol).second) {
    return nullptr;
  }
  auto med = vol->GetMedium();
  if (med && !vol->IsAssembly() && med->GetParam(1) == 0) {
    return vol;
  }
  for (int i = 0; i < vol->GetNdaughters(); ++i) {
    if (auto nf = findZeroFieldVolume(vol->GetNode(i)->GetVolume(), visited)) {
      return nf;
    }
  }
  return nullptr;
}
} // namespace

void G4LocalFieldConstruction::Construct()
{
  if (mNext) {
    mNext->Construct();
  }
  auto tree = G4UImanager::GetUIpointer()->GetTree()->FindCommandTree("/mcMagField/");
  auto field = TVirtualMC::GetMC()->GetMagField();
  if (!tree || !field) {
    return;
  }
  int nattached = 0;
  for (int i = 1; i <= tree->GetTreeEntry(); ++i) {
    std::string path = tree->GetTree(i)->GetPathName(); // "/mcMagField/<vol>/"
    auto name = path.substr(12, path.size() - 13);
    auto vol = gGeoManager->GetVolume(name.c_str());
    if (!vol) {
      LOG(warn) << "local field: no volume " << name << "; its field parameters are unused";
      continue;
    }
    // Geant4 VMC forces a local field onto all daughters, which would override zero-field media
    std::unordered_set<const TGeoVolume*> visited;
    if (auto nf = findZeroFieldVolume(vol, visited)) {
      LOG(warn) << "local field: volume " << name << " contains the zero-field volume " << nf->GetName() << "; skipped";
      continue;
    }
    vol->SetField(field);
    LOG(info) << "local field: volume " << name << " uses the parameters in /mcMagField/" << name << "/";
    ++nattached;
  }
  if (nattached > 0) {
    TG4GeometryManager::Instance()->SetIsLocalField(true);
    if (o2::conf::G4Params::Instance().navmode != o2::conf::EG4Nav::kTGeo) {
      LOG(warn) << "local field: Geant4 VMC builds local fields only with TGeo navigation; the global field applies everywhere";
    }
  }
}

} // namespace o2::g4config
