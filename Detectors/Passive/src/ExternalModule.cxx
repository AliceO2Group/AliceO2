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
#include <DetectorsBase/CADGeometryUtils.h>
#include <fstream>
#include <CommonUtils/FileSystemUtils.h>
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <TGeoMatrix.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

// ClassImp(o2::passive::ExternalModule)

namespace o2::passive
{

ExternalModule::ExternalModule(const char* name, const char* long_title, ExternalModuleOptions options) : PassiveBase(name, long_title), mOptions(options)
{
}

void ExternalModule::ConstructGeometry()
{
  // JIT the geom builder macro and obtain the top most module volume
  auto module_top = o2::base::buildCADVolumeFromMacro(mOptions.root_macro_file, GetName());
  if (!module_top) {
    LOG(error) << "No module geometry could be built from " << mOptions.root_macro_file;
    return;
  }

  // bring the CAD media under O2's MaterialManager
  o2::base::remapCADMedia(module_top, GetName());

  // place it into the provided anchor volume (needs to exist)
  auto anchor = gGeoManager->FindVolumeFast(mOptions.anchor_volume.c_str());
  if (!anchor) {
    LOG(error) << "Anchor volume " << mOptions.anchor_volume << " not found. Aborting";
    return;
  }
  anchor->AddNode(module_top, 1, const_cast<TGeoMatrix*>(mOptions.placement));
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

} // namespace o2::passive