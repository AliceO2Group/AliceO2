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

#include "Framework/ConfigParamDiscovery.h"
#include "Framework/Plugins.h"
#include <TFile.h>
#include <TMap.h>
#include <TObjString.h>

namespace o2::framework
{

void ConfigParamDiscovery::discover(ConfigParamRegistry& registry)
{
  // Keep track of the plugins
  std::vector<PluginInfo> plugins;
  if (registry.hasOption("aod-file") && registry.isSet("aod-file")) {
    auto filename = registry.get<std::string>("aod-file");
    if (filename.empty()) {
      return;
    }
    TFile* currentFile = nullptr;
    if (filename.at(0) == '@') {
      filename.erase(0, 1);
      // read the text file and set filename to the contents of the first line
      std::ifstream file(filename);
      if (!file.is_open()) {
        LOGP(fatal, "Couldn't open file \"{}\"!", filename);
      }
      std::getline(file, filename);
      file.close();
      currentFile = TFile::Open(filename.c_str());
    } else {
      currentFile = TFile::Open(filename.c_str());
    }
    if (!currentFile) {
      LOGP(fatal, "Couldn't open file \"{}\"!", filename);
    }

    // Get the metadata, if any
    auto m = (TMap*)currentFile->Get("metaData");
    if (!m) {
      LOGP(warning, "No metadata found in file \"{}\"", filename);
      return;
    }
    auto it = m->MakeIterator();

    // Serialise metadata into a ; separated string with : separating key and value
    bool first = true;
    while (auto obj = it->Next()) {
      if (first) {
        LOGP(info, "Metadata for file \"{}\":", filename);
        first = false;
      }
      auto objString = (TObjString*)m->GetValue(obj);
      LOGP(info, "- {}: {}", obj->GetName(), objString->String());
      std::string key = "aod-metadata-" + std::string(obj->GetName());
      registry.override(key.c_str(), objString->String());
    }
  }
}

} // namespace o2::framework
