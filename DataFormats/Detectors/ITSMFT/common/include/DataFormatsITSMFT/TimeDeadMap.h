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

#include <Rtypes.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace o2
{

namespace itsmft
{

class NoiseMap;

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

  void decodeMap(NoiseMap& noisemap) const;
  void decodeMap(unsigned long orbit, o2::itsmft::NoiseMap& noisemap, bool includeStaticMap = true, long orbitGapAllowed = 330000) const;
  std::string getMapVersion() const { return mMAP_VERSION; };

  unsigned long getEvolvingMapSize() const { return mEvolvingDeadMap.size(); };
  std::vector<unsigned long> getEvolvingMapKeys() const;
  void getStaticMap(std::vector<uint16_t>& mmap) const { mmap = mStaticDeadMap; };
  long getMapAtOrbit(unsigned long orbit, std::vector<uint16_t>& mmap) const;
  void setMapVersion(std::string version) { mMAP_VERSION = version; };

  bool isDefault() const { return mIsDefaultObject; };
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
