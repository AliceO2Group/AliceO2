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

#ifndef O2_FASTSIM_REGIONS_H_
#define O2_FASTSIM_REGIONS_H_

/// Deriving a model's regions from its envelope volume.
///
/// Geant4-VMC attaches a fast simulation model to regions, and in O2 a region
/// can only be "every volume of a given material" (see FastSimModel.h). To have
/// the model consulted everywhere inside a module, it must therefore be attached
/// to every material that module is built from -- which is a list nobody should
/// maintain by hand, because it changes whenever the geometry does.
///
/// So walk the envelope's subtree and collect the media as they actually are.

#include "TG4VUserPostDetConstruction.h"

#include <set>
#include <string>
#include <vector>

class TGeoVolume;

namespace o2::fastsim
{

/// Every tracking medium used by `volume` or anything below it.
/// Takes a non-const pointer because TGeo's accessors are not const.
std::set<std::string> mediaInSubtree(TGeoVolume* volume);

/// Same, looked up by volume name in the current TGeo geometry. Empty if there
/// is no such volume.
std::set<std::string> mediaInSubtree(const std::string& volumeName);

/// Sets each model's regions from its envelope, in the one window where that is
/// possible: after the geometry is built and before Geant4-VMC turns the media
/// into regions.
class FastSimRegionConstruction : public TG4VUserPostDetConstruction
{
 public:
  struct ModelRegions {
    std::string model;
    std::string envelope; ///< volume whose subtree supplies the media
    std::string regions;  ///< explicit media, used instead of the walk if given
  };

  explicit FastSimRegionConstruction(std::vector<ModelRegions> models);
  void Construct() override;

 private:
  std::vector<ModelRegions> mModels;
};

} // namespace o2::fastsim

#endif // O2_FASTSIM_REGIONS_H_
