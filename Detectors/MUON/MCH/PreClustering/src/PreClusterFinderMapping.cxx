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

#include "PreClusterFinderMapping.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <TMath.h>

#include <fairlogger/Logger.h>

#include "MCHMappingInterface/Segmentation.h"

#include "MCHPreClustering/PreClusterFinderParam.h"

namespace o2::mch
{

using namespace std;

//_________________________________________________________________________________________________
auto Mapping::addNeighbour(MpPad& pad)
{
  /// return a function to add one neighbour to a pad
  return [&pad](int neighbourID) {
    if (pad.nNeighbours == 10) {
      throw runtime_error("maximum number of neighbouring pads exceeded");
    }
    pad.neighbours[pad.nNeighbours] = neighbourID;
    ++pad.nNeighbours;
  };
}

//_________________________________________________________________________________________________
auto Mapping::addPad(MpDE& de, const mapping::Segmentation& segmentation)
{
  /// return a function to create the internal mapping for each pad
  return [&de, &segmentation](int padID) {
    MpPad& pad = de.pads[padID];
    double padX = segmentation.padPositionX(padID);
    double padY = segmentation.padPositionY(padID);
    double padSizeX = segmentation.padSizeX(padID);
    double padSizeY = segmentation.padSizeY(padID);
    pad.area[0][0] = padX - padSizeX / 2.;
    pad.area[0][1] = padX + padSizeX / 2.;
    pad.area[1][0] = padY - padSizeY / 2.;
    pad.area[1][1] = padY + padSizeY / 2.;
    pad.nNeighbours = 0;
    segmentation.forEachNeighbouringPad(padID, addNeighbour(pad));
  };
}

//_________________________________________________________________________________________________
auto Mapping::removeNeighbouringPadsInCorners(MpDE& de)
{
  /// return a function to update the neighbours of each pad, removing the ones in the corners
  /// must be called after the internal mapping is set for each pad
  return [&de](int padID) {
    auto connectedByCorners = [](float area1[2][2], float area2[2][2]) -> bool {
      constexpr float precision = -1.e-4; // overlap precision in cm: negative = decrease pad size
      return (area1[0][0] - area2[0][1] > precision || area2[0][0] - area1[0][1] > precision) &&
             (area1[1][0] - area2[1][1] > precision || area2[1][0] - area1[1][1] > precision);
    };
    MpPad& pad = de.pads[padID];
    uint8_t nSelectedNeighbours = 0;
    for (auto i = 0; i < pad.nNeighbours; ++i) {
      MpPad& neighbour = de.pads[pad.neighbours[i]];
      if (!connectedByCorners(pad.area, neighbour.area)) {
        pad.neighbours[nSelectedNeighbours] = pad.neighbours[i];
        ++nSelectedNeighbours;
      }
    }
    pad.nNeighbours = nSelectedNeighbours;
  };
}

//_________________________________________________________________________________________________
std::vector<std::unique_ptr<Mapping::MpDE>> Mapping::createMapping()
{
  /// create the internal mapping used for preclustering from the O2 mapping

  std::vector<std::unique_ptr<MpDE>> detectionElements{};

  // create the internal mapping for each DE
  mapping::forEachDetectionElement([&detectionElements](int deID) {
    auto& segmentation = mapping::segmentation(deID);
    detectionElements.push_back(std::make_unique<MpDE>());
    MpDE& de(*(detectionElements.back()));
    de.uid = deID;
    de.nPads[0] = segmentation.bending().nofPads();
    de.nPads[1] = segmentation.nonBending().nofPads();
    de.pads = std::make_unique<MpPad[]>(de.nPads[0] + de.nPads[1]);
    segmentation.forEachPad(addPad(de, segmentation));
    if (PreClusterFinderParam::Instance().excludeCorners) {
      segmentation.forEachPad(removeNeighbouringPadsInCorners(de));
    }
  });

  return detectionElements;
}

//_________________________________________________________________________________________________
bool Mapping::areOverlapping(float area1[2][2], float area2[2][2], float precision)
{
  /// check if the two areas overlap
  /// precision in cm: positive = increase pad size / negative = decrease pad size

  if (area1[0][0] - area2[0][1] > precision) {
    return false;
  }
  if (area2[0][0] - area1[0][1] > precision) {
    return false;
  }
  if (area1[1][0] - area2[1][1] > precision) {
    return false;
  }
  if (area2[1][0] - area1[1][1] > precision) {
    return false;
  }

  return true;
}

//_________________________________________________________________________________________________
bool Mapping::areOverlappingExcludeCorners(float area1[2][2], float area2[2][2])
{
  /// check if the two areas overlap (excluding pad corners)

  // precision in cm: positive = increase pad size / negative = decrease pad size
  constexpr float precision = 1.e-4;

  if (areOverlapping(area1, area2, precision)) {
    for (int ip1 = 0; ip1 < 2; ++ip1) {
      for (int ip2 = 0; ip2 < 2; ++ip2) {
        if (TMath::Abs(area1[0][ip1] - area2[0][1 - ip1]) < precision &&
            TMath::Abs(area1[1][ip2] - area2[1][1 - ip2]) < precision) {
          return false;
        }
      }
    }
    return true;
  }

  return false;
}

} // namespace o2
