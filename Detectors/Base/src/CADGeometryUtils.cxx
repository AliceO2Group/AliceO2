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

#include "DetectorsBase/CADGeometryUtils.h"
#include "DetectorsBase/MaterialManager.h"
#include <CommonUtils/FileSystemUtils.h>
#include <TGeoVolume.h>
#include <TGeoNode.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TInterpreter.h>
#include <TROOT.h>
#include <TString.h>
#include <TGlobal.h>
#include <fairlogger/Logger.h>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace o2::base
{

TGeoVolume* buildCADVolumeFromMacro(const std::string& macroFile, const std::string& instanceTag)
{
  if (macroFile.empty()) {
    return nullptr;
  }
  auto expandedHookFileName = o2::utils::expandShellVarsInFileName(macroFile);
  if (!std::filesystem::exists(expandedHookFileName)) {
    LOG(error) << "External geometry macro " << expandedHookFileName << " does not exist";
    return nullptr;
  }

  // We JIT the macro into a *unique* namespace per call. This is essential when several
  // external geometries are present at the same time: every macro produced by
  // O2_CADtoTGeo.py exports identically named symbols (build(), get_builder_hook_unchecked(),
  // LoadFacets(), ...). Loading them all into the single global Cling scope would collide
  // (the first definition wins and subsequent macros are silently ignored). By wrapping each
  // macro body in its own namespace we keep the symbols separate. The preprocessor #include
  // lines must stay at global scope, so we hoist them out of the namespace.
  std::ifstream macroStream(expandedHookFileName, std::ios::in);
  if (!macroStream.is_open()) {
    LOG(error) << "Cannot open external geometry macro " << expandedHookFileName;
    return nullptr;
  }
  std::string preamble; // #include (and other top-level preprocessor) lines -> global scope
  std::string body;     // everything else -> wrapped into a unique namespace
  std::string line;
  while (std::getline(macroStream, line)) {
    auto firstNonSpace = line.find_first_not_of(" \t");
    if (firstNonSpace != std::string::npos && line[firstNonSpace] == '#') {
      preamble += line + "\n";
    } else {
      body += line + "\n";
    }
  }

  // build a unique, valid C++ identifier for the namespace
  static std::atomic<int> instanceCounter{0};
  std::string ns = std::string("o2_cadgeom_") + instanceTag + "_" + std::to_string(instanceCounter++);
  for (auto& c : ns) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      c = '_';
    }
  }

  const std::string wrapped = preamble + "\nnamespace " + ns + " {\n" + body + "\n}\n";
  if (!gInterpreter->Declare(wrapped.c_str())) {
    LOG(error) << "Failed to JIT external geometry macro " << expandedHookFileName;
    return nullptr;
  }

  // retrieve the builder hook from the unique namespace
  const std::string globalName = "__" + ns + "_hook__";
  gROOT->ProcessLine(Form("std::function<TGeoVolume*()> %s = %s::get_builder_hook_unchecked();",
                          globalName.c_str(), ns.c_str()));
  auto global = gROOT->GetGlobal(globalName.c_str());
  if (!global) {
    LOG(error) << "Could not retrieve geometry builder hook from macro " << expandedHookFileName;
    return nullptr;
  }
  auto hook = *reinterpret_cast<std::function<TGeoVolume*()>*>(global->GetAddress());
  LOG(info) << "CAD geometry hook initialized from file " << expandedHookFileName << " (namespace " << ns << ")";

  auto top = hook();
  if (!top) {
    LOG(error) << "CAD geometry macro " << expandedHookFileName << " did not return a top volume";
  }
  return top;
}

void remapCADMedia(TGeoVolume* top, const char* modulename)
{
  std::unordered_map<TGeoMedium*, TGeoMedium*> medium_ptr_mapping;
  std::unordered_set<TGeoVolume*> volumes_already_treated;
  int counter = 1;

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
      LOG(info) << "Transforming media with name " << medium->GetName() << " for volume " << vol_->GetName();

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

  visit_volume(top);
}

} // namespace o2::base
