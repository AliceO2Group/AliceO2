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

/// @brief Flat mapping

#ifndef O2_MCH_MAPPING_H_
#define O2_MCH_MAPPING_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "MCHMappingInterface/Segmentation.h"

namespace o2
{
namespace mch
{

class Mapping
{

 public:
  // pad structure in the internal mapping
  struct MpPad {
    uint16_t iDigit;         // index of the corresponding digit
    uint8_t nNeighbours;     // number of neighbours
    uint16_t neighbours[10]; // indices of neighbours in array stored in MpDE
    float area[2][2];        // 2D area
    bool useMe;              // false if no digit attached or already visited
  };

  // DE structure in the internal mapping
  struct MpDE {
    int uid;                       // unique ID
    uint16_t nPads[2];             // number of pads on each plane
    std::unique_ptr<MpPad[]> pads; // array of pads on both planes
  };

  static std::vector<std::unique_ptr<MpDE>> createMapping();

  static bool areOverlapping(float area1[2][2], float area2[2][2], float precision);
  static bool areOverlappingExcludeCorners(float area1[2][2], float area2[2][2]);

 private:
  static auto addPad(MpDE& de, const mapping::Segmentation& segmentation);
  static auto addNeighbour(MpPad& pad);
  static auto removeNeighbouringPadsInCorners(MpDE& de);
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_MAPPING_H_
