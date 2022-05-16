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

/// \file DirectoryLoader.h
/// \brief Loading content of the Folder and returning sorted
/// \author julian.myrcha@cern.ch

#include "EventVisualisationBase/DirectoryLoader.h"
#include <filesystem>
#include <algorithm>

using namespace std;
using namespace o2::event_visualisation;

deque<string> DirectoryLoader::load(const std::string& path, const std::string& marker)
{
  deque<string> result;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ".json") {
      result.push_back(entry.path().filename());
    }
  }
  // comparison with safety if marker not in the filename (-1+1 gives 0)
  std::sort(result.begin(), result.end(),
            [marker](std::string a, std::string b) {
              return a.substr(a.find_last_of(marker) + 1) < b.substr(b.find_last_of(marker) + 1);
            });

  return result;
}
