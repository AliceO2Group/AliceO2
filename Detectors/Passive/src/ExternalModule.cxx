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
#include <fstream>
#include <atomic>
#include <cctype>
#include <CommonUtils/FileSystemUtils.h>
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <TGeoMatrix.h>
#include <TInterpreter.h>
#include <TROOT.h>
#include <TString.h>
#include <TGlobal.h>
#include <unordered_map>
#include <unordered_set>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <DetectorsBase/MaterialManager.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

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

namespace
{
// Build a TGeoCombiTrans from an optional JSON "placement" object carrying
// "translation":[x,y,z] (cm) and/or "rotation_deg":[rx,ry,rz] (deg, applied X,Y,Z).
TGeoMatrix* makePlacementFromJSON(const rapidjson::Value& placement)
{
  auto combi = new TGeoCombiTrans();
  if (placement.HasMember("rotation_deg") && placement["rotation_deg"].IsArray()) {
    const auto& r = placement["rotation_deg"];
    if (r.Size() == 3) {
      combi->RotateX(r[0].GetDouble());
      combi->RotateY(r[1].GetDouble());
      combi->RotateZ(r[2].GetDouble());
    } else {
      LOG(warning) << "ExternalModule placement 'rotation_deg' must have 3 entries; ignoring";
    }
  }
  if (placement.HasMember("translation") && placement["translation"].IsArray()) {
    const auto& t = placement["translation"];
    if (t.Size() == 3) {
      combi->SetDx(t[0].GetDouble());
      combi->SetDy(t[1].GetDouble());
      combi->SetDz(t[2].GetDouble());
    } else {
      LOG(warning) << "ExternalModule placement 'translation' must have 3 entries; ignoring";
    }
  }
  return combi;
}
} // namespace

std::vector<ExternalModule*> ExternalModule::createFromJSON(const std::string& jsonfile)
{
  std::vector<ExternalModule*> result;

  auto expanded = o2::utils::expandShellVarsInFileName(jsonfile);
  std::ifstream fileStream(expanded, std::ios::in);
  if (!fileStream.is_open()) {
    LOG(error) << "Cannot open external geometry config file '" << expanded << "'";
    return result;
  }

  rapidjson::IStreamWrapper isw(fileStream);
  rapidjson::Document doc;
  doc.ParseStream(isw);
  if (doc.HasParseError()) {
    LOG(error) << "Error parsing external geometry JSON '" << expanded << "': "
               << rapidjson::GetParseError_En(doc.GetParseError())
               << " (offset " << doc.GetErrorOffset() << ")";
    return result;
  }
  if (!doc.HasMember("externalModules") || !doc["externalModules"].IsArray()) {
    LOG(error) << "External geometry JSON '" << expanded << "' must contain an 'externalModules' array";
    return result;
  }

  auto getString = [](const rapidjson::Value& v, const char* key) -> std::string {
    if (v.HasMember(key) && v[key].IsString()) {
      return v[key].GetString();
    }
    return std::string();
  };

  for (const auto& entry : doc["externalModules"].GetArray()) {
    if (!entry.IsObject()) {
      LOG(error) << "Skipping non-object entry in 'externalModules'";
      continue;
    }
    const auto name = getString(entry, "name");
    if (name.empty()) {
      LOG(error) << "Skipping external module entry without 'name'";
      continue;
    }
    ExternalModuleOptions options;
    options.root_macro_file = getString(entry, "macro");
    options.anchor_volume = getString(entry, "anchor");
    if (options.root_macro_file.empty() || options.anchor_volume.empty()) {
      LOG(error) << "External module '" << name << "' requires both 'macro' and 'anchor'; skipping";
      continue;
    }
    if (entry.HasMember("placement") && entry["placement"].IsObject()) {
      options.placement = makePlacementFromJSON(entry["placement"]);
    }
    auto title = getString(entry, "title");
    if (title.empty()) {
      title = name;
    }
    LOG(info) << "Configured external module '" << name << "' from macro '" << options.root_macro_file
              << "' anchored to volume '" << options.anchor_volume << "'";
    result.push_back(new ExternalModule(name.c_str(), title.c_str(), options));
  }
  return result;
}

bool ExternalModule::initGeomBuilderHook()
{
  if (mOptions.root_macro_file.empty()) {
    return false;
  }
  LOG(info) << "Initializing the hook for geometry module building";
  auto expandedHookFileName = o2::utils::expandShellVarsInFileName(mOptions.root_macro_file);
  if (!std::filesystem::exists(expandedHookFileName)) {
    LOG(error) << "External geometry macro " << expandedHookFileName << " does not exist";
    return false;
  }

  // We JIT the macro into a *unique* namespace per module instance. This is essential
  // when several external modules are present at the same time: every macro produced by
  // O2_CADtoTGeo.py exports identically named symbols (build(), get_builder_hook_unchecked(),
  // LoadFacets(), ...). Loading them all into the single global Cling scope would collide
  // (the first definition wins and subsequent macros are silently ignored). By wrapping each
  // macro body in its own namespace we keep the symbols separate. The preprocessor #include
  // lines must stay at global scope, so we hoist them out of the namespace.
  std::ifstream macroStream(expandedHookFileName, std::ios::in);
  if (!macroStream.is_open()) {
    LOG(error) << "Cannot open external geometry macro " << expandedHookFileName;
    return false;
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
  std::string ns = std::string("o2_ext_") + GetName() + "_" + std::to_string(instanceCounter++);
  for (auto& c : ns) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      c = '_';
    }
  }

  const std::string wrapped = preamble + "\nnamespace " + ns + " {\n" + body + "\n}\n";
  if (!gInterpreter->Declare(wrapped.c_str())) {
    LOG(error) << "Failed to JIT external geometry macro " << expandedHookFileName;
    return false;
  }

  // retrieve the builder hook from the unique namespace
  const std::string globalName = "__" + ns + "_hook__";
  gROOT->ProcessLine(Form("std::function<TGeoVolume*()> %s = %s::get_builder_hook_unchecked();",
                          globalName.c_str(), ns.c_str()));
  auto global = gROOT->GetGlobal(globalName.c_str());
  if (!global) {
    LOG(error) << "Could not retrieve geometry builder hook from macro " << expandedHookFileName;
    return false;
  }
  mGeomHook = *reinterpret_cast<GeomBuilderFcn*>(global->GetAddress());
  LOG(info) << "Hook initialized from file " << expandedHookFileName << " (namespace " << ns << ")";
  return true;
}

} // namespace o2::passive