// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief Flat mapping

#ifndef ALICEO2_MCH_MAPPING_H_
#define ALICEO2_MCH_MAPPING_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "TExMap.h"

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
    uint8_t iCath[2];              // cathode index corresponding to each plane
    uint16_t nPads[2];             // number of pads on each plane
    std::unique_ptr<MpPad[]> pads; // array of pads on both planes
    TExMap padIndices[2];          // indices+1 of pads from their ID
  };

  static std::vector<std::unique_ptr<MpDE>> readMapping(const char* mapfile);

  static bool areOverlapping(float area1[2][2], float area2[2][2], float precision);

  static bool areOverlappingExcludeCorners(float area1[2][2], float area2[2][2]);
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_MAPPING_H_
