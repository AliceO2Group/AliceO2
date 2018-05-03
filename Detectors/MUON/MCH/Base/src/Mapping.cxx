// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHBase/Mapping.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include <TMath.h>

#include <FairMQLogger.h>

namespace o2
{
namespace mch
{

using namespace std;

std::vector<std::unique_ptr<Mapping::MpDE>> Mapping::readMapping(const char* mapFile)
{
  std::vector<std::unique_ptr<Mapping::MpDE>> detectionElements{};

  ifstream inFile(mapFile, ios::binary);

  if (!inFile.is_open()) {
    LOG(ERROR) << "Can not open " << mapFile;
    return detectionElements;
  }

  int totalNumberOfPads(0);

  int numberOfDetectionElements(0);
  inFile.read(reinterpret_cast<char*>(&numberOfDetectionElements), sizeof(numberOfDetectionElements));

  LOG(INFO) << "numberOfDetectionElements = " << numberOfDetectionElements;

  detectionElements.reserve(numberOfDetectionElements);

  for (int i = 0; i < numberOfDetectionElements; ++i) {

    detectionElements.push_back(std::make_unique<Mapping::MpDE>());
    Mapping::MpDE& de(*detectionElements[i]);

    inFile.read(reinterpret_cast<char*>(&de.uid), sizeof(de.uid));

    inFile.read(reinterpret_cast<char*>(&de.iCath[0]), sizeof(de.iCath[0]) * 2);
    inFile.read(reinterpret_cast<char*>(&de.nPads[0]), sizeof(de.nPads[0]) * 2);

    int nPadsInDE = de.nPads[0] + de.nPads[1];

    de.pads = std::make_unique<MpPad[]>(nPadsInDE);

    for (int ip = 0; ip < nPadsInDE; ++ip) {
      inFile.read(reinterpret_cast<char*>(&(de.pads[ip])), sizeof(de.pads[ip]));
      ++totalNumberOfPads;
    }

    int mapsize(2 * nPadsInDE);
    auto themap = std::make_unique<int64_t[]>(mapsize);
    int nMapElements(0);

    inFile.read(reinterpret_cast<char*>(themap.get()), sizeof(int64_t) * mapsize);

    for (int iPlane = 0; iPlane < 2; ++iPlane) {
      for (int ip = 0; ip < de.nPads[iPlane]; ++ip) {
        de.padIndices[iPlane].Add(themap[nMapElements], themap[nMapElements + 1]);
        nMapElements += 2;
      }
    }

    assert(nMapElements == 2 * nPadsInDE);
    assert(de.padIndices[0].GetSize() + de.padIndices[1].GetSize() == nPadsInDE);
  }

  if (totalNumberOfPads != 1064008 + 20992) {
    LOG(ERROR) << "totalNumberOfPads = " << totalNumberOfPads << "!= from the expected " << 1064008 + 20992;
    detectionElements.clear();
  }

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

} // namespace mch
} // namespace o2
