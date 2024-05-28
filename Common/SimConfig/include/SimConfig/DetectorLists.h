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

#ifndef O2_DETECTORLISTS_H_
#define O2_DETECTORLISTS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "Framework/Logger.h"

namespace o2::conf
{
// Container defining different general evolutions of the ALICE experiment. Each
// evolution is given a name and a list defining the names of the detectors and
// passive elements present.
using DetectorList_t = std::vector<std::string>;
using DetectorMap_t = std::unordered_map<std::string, DetectorList_t>;

// Parse the detector map from a JSON file.
// Return false if parsing failed.
bool parseDetectorMapfromJSON(const std::string& path, DetectorMap_t& map);

// Print the DetetectorMap
void printDetMap(const DetectorMap_t& map, const std::string& list = "");
} // namespace o2::conf

#endif // O2_DETECTORLISTS_H_
