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

/// \file TimeDeadMap.h
/// \brief Definition of the ITSMFT time-dependend dead map
#ifndef ALICEO2_ITSMFT_TIMEDEADMAP_H
#define ALICEO2_ITSMFT_TIMEDEADMAP_H

#include "Rtypes.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <iostream>
#include <vector>
#include <map>

namespace o2
{

namespace itsmft
{

class TimeDeadMap
{
 public:
  // Constructor
  TimeDeadMap(std::map<unsigned long, std::vector<uint16_t>>& deadmap)
  {
    mEvolvingDeadMap.swap(deadmap);
  }

  /// Constructor
  TimeDeadMap() = default;
  /// Destructor
  ~TimeDeadMap() = default;

  void fillMap(unsigned long firstOrbit, const std::vector<uint16_t>& deadVect)
  {
    mEvolvingDeadMap[firstOrbit] = deadVect;
  };

  void fillMap(const std::vector<uint16_t>& deadVect)
  {
    mStaticDeadMap = deadVect;
  }

  void clear()
  {
    mEvolvingDeadMap.clear();
    mStaticDeadMap.clear();
  }

  void decodeMap(o2::itsmft::NoiseMap& noisemap)
  { // for static part only
    if (mMAP_VERSION != "3") {
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

  void decodeMap(unsigned long orbit, o2::itsmft::NoiseMap& noisemap, bool includeStaticMap = true)
  { // for time-dependent and (optionally) static part

    if (mMAP_VERSION != "3" && mMAP_VERSION != "4") {
      LOG(error) << "Trying to decode time-dependent deadmap version " << mMAP_VERSION << ". Not implemented, doing nothing.";
      return;
    }

    if (mEvolvingDeadMap.empty()) {
      LOG(warning) << "Time-dependent dead map is empty. Doing nothing.";
      return;
    } else if (orbit > mEvolvingDeadMap.rbegin()->first + 11000 * 300 || orbit < mEvolvingDeadMap.begin()->first - 11000 * 300) {
      // the map should not leave several minutes uncovered.
      LOG(warning) << "Time-dependent dead map: the requested orbit " << orbit << " seems to be out of the range stored in the map.";
    }

    std::vector<uint16_t> closestVec;
    long dT = getMapAtOrbit(orbit, closestVec);

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
  };

  std::string getMapVersion() const { return mMAP_VERSION; };

  unsigned long getEvolvingMapSize() const { return mEvolvingDeadMap.size(); };

  std::vector<unsigned long> getEvolvingMapKeys()
  {
    std::vector<unsigned long> keys;
    std::transform(mEvolvingDeadMap.begin(), mEvolvingDeadMap.end(), std::back_inserter(keys),
                   [](const auto& O) { return O.first; });
    return keys;
  }

  void getStaticMap(std::vector<uint16_t>& mmap) { mmap = mStaticDeadMap; };

  long getMapAtOrbit(unsigned long orbit, std::vector<uint16_t>& mmap)
  { // fills mmap and returns orbit - lower_bound
    if (mEvolvingDeadMap.empty()) {
      LOG(warning) << "Requested orbit " << orbit << "from an empty time-dependent map. Doing nothing";
      return (long)orbit;
    }
    auto closest = mEvolvingDeadMap.lower_bound(orbit);
    if (closest != mEvolvingDeadMap.end()) {
      mmap = closest->second;
      return (long)orbit - closest->first;
    } else {
      mmap = mEvolvingDeadMap.rbegin()->second;
      return (long)(orbit)-mEvolvingDeadMap.rbegin()->first;
    }
  }

  void setMapVersion(std::string version) { mMAP_VERSION = version; };

  bool isDefault() { return mIsDefaultObject; };
  void setAsDefault(bool isdef = true) { mIsDefaultObject = isdef; };

 private:
  bool mIsDefaultObject = false;
  std::string mMAP_VERSION = "3";
  std::map<unsigned long, std::vector<uint16_t>> mEvolvingDeadMap; ///< Internal dead chip map representation. key = orbit
  std::vector<uint16_t> mStaticDeadMap;                            ///< To store map valid for every orbit. Filled starting from version = 4.

  ClassDefNV(TimeDeadMap, 2);
};

} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_TIMEDEADMAP_H */
