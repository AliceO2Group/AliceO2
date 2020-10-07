// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTOR_DCS_DCSPROCESSOR_H_
#define DETECTOR_DCS_DCSPROCESSOR_H_

#include <memory>
#include <Rtypes.h>
#include <unordered_map>
#include <deque>
#include "Framework/Logger.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"

/// @brief Class to process DCS data points

namespace o2
{
namespace dcs
{

class DCSProcessor
{

 public:
  using Ints = std::vector<int>;
  using Chars = std::vector<char>;
  using Doubles = std::vector<double>;
  using Binaries = std::array<uint64_t, 7>;
  using Strings = std::array<char, 56>;

  using DQChars = std::deque<char>;
  using DQInts = std::deque<int>;
  using DQDoubles = std::deque<double>;
  using DQUInts = std::deque<uint32_t>;
  using DQBools = std::deque<bool>;
  using DQStrings = std::deque<Strings>;
  using DQTimes = std::deque<uint32_t>;
  using DQBinaries = std::deque<Binaries>;

  using DPID = o2::dcs::DataPointIdentifier;
  using DPVAL = o2::dcs::DataPointValue;
  using DPCOM = o2::dcs::DataPointCompositeObject;

  DCSProcessor() = default;
  ~DCSProcessor() = default;

  void init(const std::vector<DPID>& aliaseschars, const std::vector<DPID>& aliasesints,
            const std::vector<DPID>& aliasesdoubles, const std::vector<DPID>& aliasesUints,
            const std::vector<DPID>& aliasesbools, const std::vector<DPID>& aliasesstrings,
            const std::vector<DPID>& aliasestimes, const std::vector<DPID>& aliasesbinaries);

  void init(const std::vector<DPID>& aliases);

  int process(const std::unordered_map<DPID, DPVAL>& map);

  std::unordered_map<DPID, DPVAL>::const_iterator processAlias(const DPID& alias, DeliveryType type, const std::unordered_map<DPID, DPVAL>& map);

  template <typename T>
  int processArrayType(const std::vector<DPID>& array, DeliveryType type, const std::unordered_map<DPID, DPVAL>& map, std::vector<uint64_t>& latestTimeStamp, std::unordered_map<DPID, T>& destmap);

  virtual void processChars();
  virtual void processInts();
  virtual void processDoubles();
  virtual void processUInts();
  virtual void processBools();
  virtual void processStrings();
  virtual void processTimes();
  virtual void processBinaries();
  virtual uint64_t processFlag(uint64_t flag, const char* alias);

  void doSimpleMovingAverage(int nelements, std::deque<int>& vect, float& avg, bool& isSMA);

  DQChars& getVectorForAliasChar(const DPID& id) { return mDpscharsmap[id]; }
  DQInts& getVectorForAliasInt(const DPID& id) { return mDpsintsmap[id]; }
  DQDoubles& getVectorForAliasDouble(const DPID& id) { return mDpsdoublesmap[id]; }
  DQUInts& getVectorForAliasUInt(const DPID& id) { return mDpsUintsmap[id]; }
  DQBools& getVectorForAliasBool(const DPID& id) { return mDpsboolsmap[id]; }
  DQStrings& getVectorForAliasString(const DPID& id) { return mDpsstringsmap[id]; }
  DQTimes& getVectorForAliasTime(const DPID& id) { return mDpstimesmap[id]; }
  DQBinaries& getVectorForAliasBinary(const DPID& id) { return mDpsbinariesmap[id]; }

 private:
  std::vector<float> mAvgTestInt; // moving average for DP named TestInt0
  std::unordered_map<DPID, DQChars> mDpscharsmap;
  std::unordered_map<DPID, DQInts> mDpsintsmap;
  std::unordered_map<DPID, DQDoubles> mDpsdoublesmap;
  std::unordered_map<DPID, DQUInts> mDpsUintsmap;
  std::unordered_map<DPID, DQBools> mDpsboolsmap;
  std::unordered_map<DPID, DQStrings> mDpsstringsmap;
  std::unordered_map<DPID, DQTimes> mDpstimesmap;
  std::unordered_map<DPID, DQBinaries> mDpsbinariesmap;
  std::vector<DPID> mAliaseschars;
  std::vector<DPID> mAliasesints;
  std::vector<DPID> mAliasesdoubles;
  std::vector<DPID> mAliasesUints;
  std::vector<DPID> mAliasesbools;
  std::vector<DPID> mAliasesstrings;
  std::vector<DPID> mAliasestimes;
  std::vector<DPID> mAliasesbinaries;
  std::vector<uint64_t> mLatestTimestampchars;
  std::vector<uint64_t> mLatestTimestampints;
  std::vector<uint64_t> mLatestTimestampdoubles;
  std::vector<uint64_t> mLatestTimestampUints;
  std::vector<uint64_t> mLatestTimestampbools;
  std::vector<uint64_t> mLatestTimestampstrings;
  std::vector<uint64_t> mLatestTimestamptimes;
  std::vector<uint64_t> mLatestTimestampbinaries;

  ClassDefNV(DCSProcessor, 0);
};

using Ints = std::vector<int>;
using Chars = std::vector<char>;
using Doubles = std::vector<double>;
using Binaries = std::array<uint64_t, 7>;
using Strings = std::array<char, 56>;

using DQStrings = std::deque<Strings>;
using DQBinaries = std::deque<Binaries>;

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

template <typename T>
int DCSProcessor::processArrayType(const std::vector<DPID>& array, DeliveryType type, const std::unordered_map<DPID, DPVAL>& map, std::vector<uint64_t>& latestTimeStamp, std::unordered_map<DPID, T>& destmap)
{

  // processing the array of type T

  int found = 0;
  auto s = array.size();
  if (s > 0) {
    for (size_t i = 0; i != s; ++i) {
      auto it = processAlias(array[i], type, map);
      if (it == map.end()) {
        LOG(ERROR) << "Element " << array[i] << " not found " << std::endl;
        continue;
      }
      found++;
      auto& val = it->second;
      auto flags = val.get_flags();
      if (processFlag(flags, array[i].get_alias()) == 0) {
        auto etime = val.get_epoch_time();
        // fill only if new value has a timestamp different from the timestamp of the previous one
        LOG(DEBUG) << "destmap[array[" << i << "]].size() = " << destmap[array[i]].size();
        if (destmap[array[i]].size() == 0 || etime != latestTimeStamp[i]) {
          LOG(DEBUG) << "adding new value";
          destmap[array[i]].push_back(val.payload_pt1);
          latestTimeStamp[i] = etime;
        }
      }
    }
  }
  return found;
}

template <>
int DCSProcessor::processArrayType(const std::vector<DCSProcessor::DPID>& array, DeliveryType type, const std::unordered_map<DCSProcessor::DPID, DCSProcessor::DPVAL>& map, std::vector<uint64_t>& latestTimeStamp, std::unordered_map<DCSProcessor::DPID, DCSProcessor::DQStrings>& destmap);

template <>
int DCSProcessor::processArrayType(const std::vector<DCSProcessor::DPID>& array, DeliveryType type, const std::unordered_map<DCSProcessor::DPID, DCSProcessor::DPVAL>& map, std::vector<uint64_t>& latestTimeStamp, std::unordered_map<DCSProcessor::DPID, DCSProcessor::DQBinaries>& destmap);

} // namespace dcs
} // namespace o2

#endif
