// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Sandro Wenzel (CERN), 2026

#include <DetectorsPassive/ExternalModule.h>
#include <CommonUtils/ConfigurationMacroHelper.h>
#include <filesystem>
#include <CommonUtils/FileSystemUtils.h>
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <unordered_map>
#include <unordered_set>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <DetectorsBase/MaterialManager.h>

// ClassImp(o2::passive::ExternalModule)

namespace o2::passive
{

ExternalModule::ExternalModule(const char* name, const char* long_title, ExternalModuleOptions options) : PassiveBase(name, long_title), mOptions(options)
{
}

void ExternalModule::remapMedia(TGeoVolume* top_volume)
{
  std::unordered_map<TGeoMedium*, TGeoMedium*> medium_ptr_mapping;
  std::unordered_set<TGeoVolume*> volumes_already_treated;
  int counter = 1;

  auto modulename = GetName();

  // The transformer function
  auto transform_media = [&](TGeoVolume* vol_) {
    if (volumes_already_treated.find(vol_) != volumes_already_treated.end()) {
      // this volume was already transformed
      return;
    }
    volumes_already_treated.insert(vol_);

    if (dynamic_cast<TGeoVolumeAssembly*>(vol_)) {
      // do nothing for assemblies (they don't have a medium)
      return;
    }

    auto medium = vol_->GetMedium();
    if (!medium) {
      return;
    }

    auto iter = medium_ptr_mapping.find(medium);
    if (iter != medium_ptr_mapping.end()) {
      // This medium has already been transformed, so
      // we just update the volume
      vol_->SetMedium(iter->second);
      return;
    } else {
      std::cout << "Transforming media with name " << medium->GetName() << " for volume " << vol_->GetName() << "\n";

      // we found a medium, not yet treated
      auto curr_mat = medium->GetMaterial();
      auto& matmgr = o2::base::MaterialManager::Instance();

      matmgr.Material(modulename, counter, curr_mat->GetName(), curr_mat->GetA(), curr_mat->GetZ(), curr_mat->GetDensity(), curr_mat->GetRadLen(), curr_mat->GetIntLen());
      // TGeo medium params are stored in a flat array with the following convention
      // fParams[0] = isvol;
      // fParams[1] = ifield;
      // fParams[2] = fieldm;
      // fParams[3] = tmaxfd;
      // fParams[4] = stemax;
      // fParams[5] = deemax;
      // fParams[6] = epsil;
      // fParams[7] = stmin;
      const auto isvol = medium->GetParam(0);
      const auto isxfld = medium->GetParam(1);
      const auto sxmgmx = medium->GetParam(2);
      const auto tmaxfd = medium->GetParam(3);
      const auto stemax = medium->GetParam(4);
      const auto deemax = medium->GetParam(5);
      const auto epsil = medium->GetParam(6);
      const auto stmin = medium->GetParam(7);

      matmgr.Medium(modulename, counter, medium->GetName(), counter, isvol, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

      // there will be new Material and Medium objects; fetch them
      auto new_med = matmgr.getTGeoMedium(modulename, counter);

      // insert into cache
      medium_ptr_mapping[medium] = new_med;
      vol_->SetMedium(new_med);
      counter++;
    }
  }; // end transformer lambda

  // a generic volume walker
  std::function<void(TGeoVolume*)> visit_volume;
  visit_volume = [&](TGeoVolume* vol) -> void {
    if (!vol) {
      return;
    }

    // call the transformer
    transform_media(vol);

    // Recurse into daughters
    const int nd = vol->GetNdaughters();
    for (int i = 0; i < nd; ++i) {
      TGeoNode* node = vol->GetNode(i);
      if (!node) {
        continue;
      }
      TGeoVolume* child = node->GetVolume();
      if (!child) {
        continue;
      }

      visit_volume(child);
    }
  };

  visit_volume(top_volume);
}

void ExternalModule::ConstructGeometry()
{
  // JIT the geom builder hook
  if (!initGeomBuilderHook()) {
    LOG(error) << " Could not load geometry builder hook";
    return;
  }

  // otherwise execute it and obtain pointer to top most module volume
  auto module_top = mGeomHook();
  if (!module_top) {
    LOG(error) << "No module found\n";
    return;
  }

  remapMedia(const_cast<TGeoVolume*>(module_top));

  // place it into the provided anchor volume (needs to exist)
  auto anchor = gGeoManager->FindVolumeFast(mOptions.anchor_volume.c_str());
  if (!anchor) {
    LOG(error) << "Anchor volume " << mOptions.anchor_volume << " not found. Aborting";
    return;
  }
  anchor->AddNode(const_cast<TGeoVolume*>(module_top), 1, const_cast<TGeoMatrix*>(mOptions.placement));
}

bool ExternalModule::initGeomBuilderHook()
{
  if (mOptions.root_macro_file.size() > 0) {
    LOG(info) << "Initializing the hook for geometry module building";
    auto expandedHookFileName = o2::utils::expandShellVarsInFileName(mOptions.root_macro_file);
    if (std::filesystem::exists(expandedHookFileName)) {
      // if this file exists we will compile the hook on the fly (the last one is an identifier --> maybe make it dependent on this class)
      mGeomHook = o2::conf::GetFromMacro<GeomBuilderFcn>(mOptions.root_macro_file, "get_builder_hook_unchecked()", "function<TGeoVolume*()>", "o2_passive_extmodule_builder");
      LOG(info) << "Hook initialized from file " << expandedHookFileName;
      return true;
    }
  }
  return false;
}

} // namespace o2::passive