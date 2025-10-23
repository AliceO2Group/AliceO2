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

/// \file TimeDeadMap.cxx
/// \brief Implementation of the time-dependent map

#include "DataFormatsITSMFT/TimeDeadMap.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "Framework/Logger.h"

using namespace o2::itsmft;

void TimeDeadMap::decodeMap(o2::itsmft::NoiseMap& noisemap) const
{ // for static part only
  if (mMAP_VERSION == "3") {
    LOG(error) << "Trying to decode static part of deadmap version " << mMAP_VERSION << ". Not implemented, doing nothing.";
    return;
  }
  for (int iel = 0; iel < mStaticDeadMap.size(); iel++) {
    uint16_t w = mStaticDeadMap[iel];
    noisemap.maskFullChip(w & 0x7FFF);
    if (w & 0x8000) {
      for (int w2 = (w & 0x7FFF) + 1; w2 < mStaticDeadMap.at(iel + 1); w2++) {
        noisemap.maskFullChip(w2);
      }
    }
  }
}

void TimeDeadMap::decodeMap(unsigned long orbit, o2::itsmft::NoiseMap& noisemap, bool includeStaticMap, long orbitGapAllowed) const
{ // for time-dependent and (optionally) static part. Use orbitGapAllowed = -1 to ignore check on orbit difference

  if (mMAP_VERSION != "3" && mMAP_VERSION != "4") {
    LOG(error) << "Trying to decode time-dependent deadmap version " << mMAP_VERSION << ". Not implemented, doing nothing.";
    return;
  }

  if (mEvolvingDeadMap.empty()) {
    LOG(warning) << "Time-dependent dead map is empty. Doing nothing.";
    return;
  }

  std::vector<uint16_t> closestVec;
  long dT = getMapAtOrbit(orbit, closestVec);

  if (orbitGapAllowed >= 0 && std::abs(dT) > orbitGapAllowed) {
    LOG(warning) << "Requested orbit " << orbit << ", found " << orbit - dT << ". Orbit gap is too high, skipping time-dependent map.";
    closestVec.clear();
  }

  // add static part if requested. something may be masked twice
  if (includeStaticMap && mMAP_VERSION != "3") {
    closestVec.insert(closestVec.end(), mStaticDeadMap.begin(), mStaticDeadMap.end());
  }

  // vector encoding: if 1<<15 = 0x8000 is set, the word encodes the first element of a range, with mask (1<<15)-1 = 0x7FFF. The last element of the range is the next in the vector.

  for (int iel = 0; iel < closestVec.size(); iel++) {
    uint16_t w = closestVec.at(iel);
    noisemap.maskFullChip(w & 0x7FFF);
    if (w & 0x8000) {
      for (int w2 = (w & 0x7FFF) + 1; w2 < closestVec.at(iel + 1); w2++) {
        noisemap.maskFullChip(w2);
      }
    }
  }
}

std::vector<unsigned long> TimeDeadMap::getEvolvingMapKeys() const
{
  std::vector<unsigned long> keys;
  std::transform(mEvolvingDeadMap.begin(), mEvolvingDeadMap.end(), std::back_inserter(keys),
                 [](const auto& O) { return O.first; });
  return keys;
}

long TimeDeadMap::getMapAtOrbit(unsigned long orbit, std::vector<uint16_t>& mmap) const
{ // fills mmap and returns requested_orbit - found_orbit. Found orbit is the highest key lower or equal to the requested one
  if (mEvolvingDeadMap.empty()) {
    LOG(warning) << "Requested orbit " << orbit << "from an empty time-dependent map. Doing nothing";
    return (long)orbit;
  }
  auto closest = mEvolvingDeadMap.upper_bound(orbit);
  if (closest != mEvolvingDeadMap.begin()) {
    --closest;
    mmap = closest->second;
    return (long)orbit - closest->first;
  } else {
    mmap = mEvolvingDeadMap.begin()->second;
    return (long)(orbit)-mEvolvingDeadMap.begin()->first;
  }
}
