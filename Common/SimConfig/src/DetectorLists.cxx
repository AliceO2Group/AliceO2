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

#include "SimConfig/DetectorLists.h"
#include <fairlogger/Logger.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>

namespace o2::conf
{

bool parseDetectorMapfromJSON(const std::string& path, DetectorMap_t& map)
{
  // Parse JSON file to build map
  std::ifstream fileStream(path, std::ios::in);
  if (!fileStream.is_open()) {
    LOGP(error, "Cannot open '{}'!", path);
    return false;
  }
  rapidjson::IStreamWrapper isw(fileStream);
  rapidjson::Document doc;
  doc.ParseStream(isw);
  if (doc.HasParseError()) {
    LOGP(error, "Error parsing provided json file '{}':", path);
    LOGP(error, "  - Error -> {}", rapidjson::GetParseError_En(doc.GetParseError()));
    LOGP(error, "  - Offset -> {}", doc.GetErrorOffset());
    return false;
  }

  // Clear and rebuild map
  map.clear();
  try {
    for (auto verItr = doc.MemberBegin(); verItr != doc.MemberEnd(); ++verItr) {
      const auto& version = verItr->name.GetString();
      DetectorList_t list;
      const auto& elements = doc[version];
      for (const auto& ele : elements.GetArray()) {
        list.emplace_back(ele.GetString());
      }
      map.emplace(version, list);
    }
  } catch (const std::exception& e) {
    LOGP(error, "Failed to build detector map from file '{}' with '{}'", path, e.what());
    return false;
  }

  return true;
}

void printDetMap(const DetectorMap_t& map, const std::string& list)
{
  if (list.empty()) {
    LOGP(error, "List of all available versions including their detectors:");
    for (int i{0}; const auto& [version, elements] : map) {
      LOGP(error, " - {: >2d}. {}:", i++, version);
      for (int j{0}; const auto& element : elements) {
        LOGP(error, "\t\t* {: >2d}.\t{}", j++, element);
      }
    }
  } else {
    LOGP(error, "List of available modules for version {}:", list);
    for (int j{0}; const auto& element : map.at(list)) {
      LOGP(error, "\t* {: >2d}.\t{}", j++, element);
    }
  }
}

} // namespace o2::conf
