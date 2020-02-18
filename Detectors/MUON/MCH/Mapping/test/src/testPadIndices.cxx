// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "InputDocument.h"
#include <string>
#include <iostream>
#include <fmt/format.h>
#include "MCHMappingInterface/Segmentation.h"

int CheckPadIndexIsAsExpected(std::string filepath)
{
  InputWrapper data(filepath.c_str());

  rapidjson::Value& padIndices = data.document()["channels"];

  int nbad{0};

  for (auto& tp : padIndices.GetArray()) {
    int detElemId = tp["de"].GetInt();
    bool isBendingPlane = (tp["bending"].GetString() == std::string("true"));
    int refPadIndex = tp["padindex"].GetInt();
    int bpad;
    int nbpad;
    o2::mch::mapping::segmentation(detElemId).findPadPairByPosition(tp["x"].GetDouble(), tp["y"].GetDouble(), bpad, nbpad);
    int padIndex = (isBendingPlane ? bpad : nbpad);
    if (padIndex != refPadIndex) {
      ++nbad;
      std::cout << fmt::format("Expected index = {} but got {}\n", refPadIndex, padIndex);
    }
  }

  return nbad;
}

int main(int argc, char** argv)
{
  std::string filepath;
  bool manunumbering{false};

  if (argc >= 1) {
    std::string padindices{"--padindices"};
    std::string testnumbering{"--manunumbering"};
    for (auto i = 0; i < argc; i++) {
      if (padindices == argv[i] && i < argc - 1) {
        filepath = argv[i + 1];
        ++i;
      }
      if (testnumbering == argv[i]) {
        manunumbering = true;
      }
    }
  }

  return CheckPadIndexIsAsExpected(filepath);
}
