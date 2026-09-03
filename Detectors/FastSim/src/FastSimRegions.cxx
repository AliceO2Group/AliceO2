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

#include "FastSim/FastSimRegions.h"

#include "TG4GeometryManager.h"
#include "TG4ModelConfigurationManager.h"

#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>

#include <fairlogger/Logger.h>

namespace o2::fastsim
{

namespace
{
void collect(TGeoVolume* volume, std::set<std::string>& media, std::set<TGeoVolume*>& seen)
{
  if (volume == nullptr || !seen.insert(volume).second) {
    return; // a volume can be placed many times; visit it once
  }
  if (const TGeoMedium* medium = volume->GetMedium()) {
    media.insert(medium->GetName());
  }
  TObjArray* nodes = volume->GetNodes();
  if (nodes == nullptr) {
    return;
  }
  for (int i = 0; i < nodes->GetEntriesFast(); ++i) {
    auto* node = static_cast<TGeoNode*>(nodes->UncheckedAt(i));
    if (node != nullptr) {
      collect(node->GetVolume(), media, seen);
    }
  }
}
} // namespace

//_____________________________________________________________________________
std::set<std::string> mediaInSubtree(TGeoVolume* volume)
{
  std::set<std::string> media;
  std::set<TGeoVolume*> seen;
  collect(volume, media, seen);
  return media;
}

//_____________________________________________________________________________
std::set<std::string> mediaInSubtree(const std::string& volumeName)
{
  if (gGeoManager == nullptr) {
    LOG(error) << "fast simulation: no TGeo geometry when resolving '" << volumeName << "'";
    return {};
  }
  auto* volume = gGeoManager->GetVolume(volumeName.c_str());
  if (volume == nullptr) {
    LOG(error) << "fast simulation: no volume named '" << volumeName << "' in the geometry";
    return {};
  }
  return mediaInSubtree(volume);
}

//_____________________________________________________________________________
FastSimRegionConstruction::FastSimRegionConstruction(std::vector<ModelRegions> models)
  : mModels(std::move(models))
{
}

//_____________________________________________________________________________
void FastSimRegionConstruction::Construct()
{
  /// Called by Geant4-VMC after the geometry is built and immediately before
  /// the media are turned into regions, which is the only moment at which this
  /// can be done: the geometry does not exist when the model is created, and
  /// the regions are fixed once they are made.
  auto* manager = TG4GeometryManager::Instance()->GetFastModelsManager();

  for (const auto& model : mModels) {
    std::string regions = model.regions;
    if (regions.empty()) {
      const auto media = mediaInSubtree(model.envelope);
      for (const auto& medium : media) {
        // The setter tokenizes on whitespace, so a medium whose name contains a
        // space cannot go through it. O2 composes medium names as
        // <MODULE>_<name> and none contain spaces, but say so if that changes
        // rather than silently selecting the wrong thing.
        if (medium.find(' ') != std::string::npos) {
          LOG(warn) << "fast simulation: medium '" << medium << "' contains a space and cannot "
                    << "be selected; it is skipped";
          continue;
        }
        regions += (regions.empty() ? "" : " ") + medium;
      }
      LOG(info) << "fast simulation: model " << model.model << " covers " << media.size()
                << " media found under '" << model.envelope << "'";
    }
    if (regions.empty()) {
      LOG(error) << "fast simulation: model " << model.model << " ended up with no regions";
      continue;
    }
    LOG(debug) << "fast simulation: regions for " << model.model << ": " << regions;
    manager->SetModelRegions(model.model, regions);
  }
}

} // namespace o2::fastsim
