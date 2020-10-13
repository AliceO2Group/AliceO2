// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <DetectorsDCS/DCSProcessor.h>
#include "Rtypes.h"
#include <deque>
#include <string>
#include <algorithm>
#include <iterator>

using namespace o2::dcs;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

//ClassImp(o2::dcs::DCSProcessor);

void DCSProcessor::init(const std::vector<DPID>& aliaseschars, const std::vector<DPID>& aliasesints, const std::vector<DPID>& aliasesdoubles,
                        const std::vector<DPID>& aliasesUints, const std::vector<DPID>& aliasesbools, const std::vector<DPID>& aliasesstrings,
                        const std::vector<DPID>& aliasestimes, const std::vector<DPID>& aliasesbinaries)
{

  // init from separate vectors of aliases (one per data point type)

  // chars
  for (auto it = std::begin(aliaseschars); it != std::end(aliaseschars); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_CHAR) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a char";
    }
    mAliaseschars.emplace_back((*it).get_alias(), DeliveryType::RAW_CHAR);
  }

  // ints
  for (auto it = std::begin(aliasesints); it != std::end(aliasesints); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_INT) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a int";
    }
    mAliasesints.emplace_back((*it).get_alias(), DeliveryType::RAW_INT);
  }

  // doubles
  for (auto it = std::begin(aliasesdoubles); it != std::end(aliasesdoubles); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_DOUBLE) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a double";
    }
    mAliasesdoubles.emplace_back((*it).get_alias(), DeliveryType::RAW_DOUBLE);
  }

  // uints
  for (auto it = std::begin(aliasesUints); it != std::end(aliasesUints); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_UINT) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a uint";
    }
    mAliasesUints.emplace_back((*it).get_alias(), DeliveryType::RAW_UINT);
  }

  // bools
  for (auto it = std::begin(aliasesbools); it != std::end(aliasesbools); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_BOOL) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a bool";
    }
    mAliasesbools.emplace_back((*it).get_alias(), DeliveryType::RAW_BOOL);
  }

  // strings
  for (auto it = std::begin(aliasesstrings); it != std::end(aliasesstrings); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_STRING) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a string";
    }
    mAliasesstrings.emplace_back((*it).get_alias(), DeliveryType::RAW_STRING);
  }

  // times
  for (auto it = std::begin(aliasestimes); it != std::end(aliasestimes); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_TIME) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a time";
    }
    mAliasestimes.emplace_back((*it).get_alias(), DeliveryType::RAW_TIME);
  }

  // binaries
  for (auto it = std::begin(aliasesbinaries); it != std::end(aliasesbinaries); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_BINARY) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a binary";
    }
    mAliasesbinaries.emplace_back((*it).get_alias(), DeliveryType::RAW_BINARY);
  }

  mLatestTimestampchars.resize(aliaseschars.size(), 0);
  mLatestTimestampints.resize(aliasesints.size(), 0);
  mLatestTimestampdoubles.resize(aliasesdoubles.size(), 0);
  mLatestTimestampUints.resize(aliasesUints.size(), 0);
  mLatestTimestampbools.resize(aliasesbools.size(), 0);
  mLatestTimestampstrings.resize(aliasesstrings.size(), 0);
  mLatestTimestamptimes.resize(aliasestimes.size(), 0);
  mLatestTimestampbinaries.resize(aliasesbinaries.size(), 0);
}

//______________________________________________________________________

void DCSProcessor::init(const std::vector<DPID>& aliases)
{

  int nchars = 0, nints = 0, ndoubles = 0, nUints = 0,
      nbools = 0, nstrings = 0, ntimes = 0, nbinaries = 0;
  for (auto it = std::begin(aliases); it != std::end(aliases); ++it) {
    if ((*it).get_type() == DeliveryType::RAW_CHAR) {
      mAliaseschars.emplace_back((*it).get_alias(), DeliveryType::RAW_CHAR);
      nchars++;
    }
    if ((*it).get_type() == DeliveryType::RAW_INT) {
      mAliasesints.emplace_back((*it).get_alias(), DeliveryType::RAW_INT);
      nints++;
    }
    if ((*it).get_type() == DeliveryType::RAW_DOUBLE) {
      mAliasesdoubles.emplace_back((*it).get_alias(), DeliveryType::RAW_DOUBLE);
      ndoubles++;
    }
    if ((*it).get_type() == DeliveryType::RAW_UINT) {
      mAliasesUints.emplace_back((*it).get_alias(), DeliveryType::RAW_UINT);
      nUints++;
    }
    if ((*it).get_type() == DeliveryType::RAW_BOOL) {
      mAliasesbools.emplace_back((*it).get_alias(), DeliveryType::RAW_BOOL);
      nbools++;
    }
    if ((*it).get_type() == DeliveryType::RAW_STRING) {
      mAliasesstrings.emplace_back((*it).get_alias(), DeliveryType::RAW_STRING);
      nstrings++;
    }
    if ((*it).get_type() == DeliveryType::RAW_TIME) {
      mAliasestimes.emplace_back((*it).get_alias(), DeliveryType::RAW_TIME);
      ntimes++;
    }
    if ((*it).get_type() == DeliveryType::RAW_BINARY) {
      mAliasesbinaries.emplace_back((*it).get_alias(), DeliveryType::RAW_BINARY);
      nbinaries++;
    }
  }

  mLatestTimestampchars.resize(nchars, 0);
  mLatestTimestampints.resize(nints, 0);
  mLatestTimestampdoubles.resize(ndoubles, 0);
  mLatestTimestampUints.resize(nUints, 0);
  mLatestTimestampbools.resize(nbools, 0);
  mLatestTimestampstrings.resize(nstrings, 0);
  mLatestTimestamptimes.resize(ntimes, 0);
  mLatestTimestampbinaries.resize(nbinaries, 0);
}

//__________________________________________________________________

int DCSProcessor::processMap(const std::unordered_map<DPID, DPVAL>& map, bool isDelta)
{

  // process function to do "something" with the DCS map that is passed

  // resetting the content of the CCDB object to be sent

  if (!isDelta) {
    // full map sent
    mFullMapSent = true;
  } else {
    if (!mFullMapSent) {
      LOG(ERROR) << "We need first a full map!";
    }
    mNCyclesNoFullMap++;
    if (mNCyclesNoFullMap > mMaxCyclesNoFullMap) {
      LOG(ERROR) << "We expected a full map!";
    }
  }

  mIsDelta = isDelta;

  // we need to check if there are the Data Points that we need

  int foundChars = 0, foundInts = 0, foundDoubles = 0, foundUInts = 0,
      foundBools = 0, foundStrings = 0, foundTimes = 0, foundBinaries = 0;

  // char type
  foundChars = processArrayType(mAliaseschars, DeliveryType::RAW_CHAR, map, mLatestTimestampchars, mDpscharsmap);
  if (foundChars > 0)
    processChars();

  // int type
  foundInts = processArrayType(mAliasesints, DeliveryType::RAW_INT, map, mLatestTimestampints, mDpsintsmap);

  // double type
  foundDoubles = processArrayType(mAliasesdoubles, DeliveryType::RAW_DOUBLE, map, mLatestTimestampdoubles, mDpsdoublesmap);

  // UInt type
  foundUInts = processArrayType(mAliasesUints, DeliveryType::RAW_UINT, map, mLatestTimestampUints, mDpsUintsmap);

  // Bool type
  foundBools = processArrayType(mAliasesbools, DeliveryType::RAW_BOOL, map, mLatestTimestampbools, mDpsboolsmap);

  // String type
  foundStrings = processArrayType(mAliasesstrings, DeliveryType::RAW_STRING, map, mLatestTimestampstrings, mDpsstringsmap);

  // Time type
  foundTimes = processArrayType(mAliasestimes, DeliveryType::RAW_TIME, map, mLatestTimestamptimes, mDpstimesmap);

  // Binary type
  foundBinaries = processArrayType(mAliasesbinaries, DeliveryType::RAW_BINARY, map, mLatestTimestampbinaries, mDpsbinariesmap);

  if (!isDelta) {
    if (foundChars != mAliaseschars.size())
      LOG(INFO) << "Not all expected char-typed DPs found!";
    if (foundInts != mAliasesints.size())
      LOG(INFO) << "Not all expected int-typed DPs found!";
    if (foundDoubles != mAliasesdoubles.size())
      LOG(INFO) << "Not all expected double-typed DPs found!";
    if (foundUInts != mAliasesUints.size())
      LOG(INFO) << "Not all expected uint-typed DPs found!";
    if (foundBools != mAliasesbools.size())
      LOG(INFO) << "Not all expected bool-typed DPs found!";
    if (foundStrings != mAliasesstrings.size())
      LOG(INFO) << "Not all expected string-typed DPs found!";
    if (foundTimes != mAliasestimes.size())
      LOG(INFO) << "Not all expected time-typed DPs found!";
    if (foundBinaries != mAliasesbinaries.size())
      LOG(INFO) << "Not all expected binary-typed DPs found!";
  }

  // filling CCDB info to be sent in output
  std::map<std::string, std::string> md;
  prepareCCDBobject(mccdbSimpleMovingAverage, mccdbSimpleMovingAverageInfo, "TestDCS/SimpleMovingAverageDPs", mTF, md);

  LOG(DEBUG) << "Size of unordered_map for CCDB = " << mccdbSimpleMovingAverage.size();
  LOG(DEBUG) << "CCDB entry for TF " << mTF << " will be:";
  for (const auto& i : mccdbSimpleMovingAverage) {
    LOG(DEBUG) << i.first << " --> " << i.second;
  }

  return 0;
}

//__________________________________________________________________

int DCSProcessor::processDP(const std::pair<DPID, DPVAL>& dpcom)
{

  // processing single DP

  DPID dpid = dpcom.first;
  DeliveryType type = dpid.get_type();

  // first we check if the DP is in the list for the detector
  if (type == DeliveryType::RAW_CHAR) {
    auto it = std::find(mAliaseschars.begin(), mAliaseschars.end(), dpid);
    if (it == mAliaseschars.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliaseschars.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestampchars[index], mDpscharsmap);
    process(dpid, mDpscharsmap[dpid]);
  }

  else if (type == DeliveryType::RAW_INT) {
    auto it = std::find(mAliasesints.begin(), mAliasesints.end(), dpid);
    if (it == mAliasesints.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliasesints.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestampints[index], mDpsintsmap);
    process(dpid, mDpsintsmap[dpid]);
  }

  else if (type == DeliveryType::RAW_DOUBLE) {
    auto it = std::find(mAliasesdoubles.begin(), mAliasesdoubles.end(), dpid);
    if (it == mAliasesdoubles.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliasesdoubles.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestampdoubles[index], mDpsdoublesmap);
    process(dpid, mDpsdoublesmap[dpid]);
  }

  else if (type == DeliveryType::RAW_UINT) {
    auto it = std::find(mAliasesUints.begin(), mAliasesUints.end(), dpid);
    if (it == mAliasesUints.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliasesUints.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestampUints[index], mDpsUintsmap);
    process(dpid, mDpsUintsmap[dpid]);
  }

  else if (type == DeliveryType::RAW_BOOL) {
    auto it = std::find(mAliasesbools.begin(), mAliasesbools.end(), dpid);
    if (it == mAliasesbools.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliasesbools.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestampbools[index], mDpsboolsmap);
    process(dpid, mDpsboolsmap[dpid]);
  }

  else if (type == DeliveryType::RAW_STRING) {
    auto it = std::find(mAliasesstrings.begin(), mAliasesstrings.end(), dpid);
    if (it == mAliasesstrings.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliasesstrings.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestampstrings[index], mDpsstringsmap);
    process(dpid, mDpsstringsmap[dpid]);
  } else if (type == DeliveryType::RAW_TIME) {
    auto it = std::find(mAliasestimes.begin(), mAliasestimes.end(), dpid);
    if (it == mAliasestimes.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliasestimes.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestamptimes[index], mDpstimesmap);
    process(dpid, mDpstimesmap[dpid]);
  } else if (type == DeliveryType::RAW_BINARY) {
    auto it = std::find(mAliasesbinaries.begin(), mAliasesbinaries.end(), dpid);
    if (it == mAliasesbinaries.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    int index = std::distance(mAliasesbinaries.begin(), it);
    checkFlagsAndFill(dpcom, mLatestTimestampbinaries[index], mDpsbinariesmap);
    process(dpid, mDpsbinariesmap[dpid]);
  }

  return 0;
}

//______________________________________________________________________

template <>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, int64_t& latestTimeStamp, std::unordered_map<DPID, DQStrings>& destmap)
{

  // check the flags for the upcoming data, and if ok, fill the accumulator

  auto& dpid = dpcom.first;
  auto& val = dpcom.second;
  auto flags = val.get_flags();
  if (processFlag(flags, dpid.get_alias()) == 0) {
    auto etime = val.get_epoch_time();
    // fill only if new value has a timestamp different from the timestamp of the previous one
    LOG(DEBUG) << "destmap[pid].size() = " << destmap[dpid].size();
    if (destmap[dpid].size() == 0 || etime != std::abs(latestTimeStamp)) {
      auto& tmp = destmap[dpid].emplace_back();
      std::strncpy(tmp.data(), (char*)&(val.payload_pt1), 56);
      latestTimeStamp = etime;
    }
  }
}

//______________________________________________________________________

template <>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, int64_t& latestTimeStamp, std::unordered_map<DPID, DQBinaries>& destmap)
{

  // check the flags for the upcoming data, and if ok, fill the accumulator

  auto& dpid = dpcom.first;
  auto& val = dpcom.second;
  auto flags = val.get_flags();
  if (processFlag(flags, dpid.get_alias()) == 0) {
    auto etime = val.get_epoch_time();
    // fill only if new value has a timestamp different from the timestamp of the previous one
    LOG(DEBUG) << "destmap[pid].size() = " << destmap[dpid].size();
    if (destmap[dpid].size() == 0 || etime != std::abs(latestTimeStamp)) {
      auto& tmp = destmap[dpid].emplace_back();
      memcpy(tmp.data(), &(val.payload_pt1), 7);
      latestTimeStamp = etime;
    }
  }
}

//______________________________________________________________________

template <>
void DCSProcessor::process(const DPID& alias, std::deque<int>& aliasdeque)
{
  // processing the single alias of type int
  bool isSMA = false;
  doSimpleMovingAverage(2, aliasdeque, mSimpleMovingAverage[alias], isSMA);
  LOG(DEBUG) << "Moving average = " << mSimpleMovingAverage[alias];
  // create CCDB object
  //if (isSMA) {
  mccdbSimpleMovingAverage[alias.get_alias()] = mSimpleMovingAverage[alias];
  //}
  return;
}

//______________________________________________________________________

std::unordered_map<DPID, DPVAL>::const_iterator DCSProcessor::findAndCheckAlias(const DPID& alias, DeliveryType type, const std::unordered_map<DPID, DPVAL>& map)
{

  // processing basic checks for map: all needed aliases must be present
  // finds dp defined by "alias" in received map "map"

  LOG(DEBUG) << "Processing " << alias;
  auto it = map.find(alias);
  DeliveryType tt = alias.get_type();
  if (tt != type) {
    LOG(FATAL) << "Delivery Type of alias " << alias.get_alias() << " does not match definition in DCSProcessor (" << type << ")! Please fix";
  }
  return it;
}

//______________________________________________________________________

void DCSProcessor::processChars()
{

  // function to process aliases of Char type; it will just print them

  for (size_t i = 0; i != mAliaseschars.size(); ++i) {
    LOG(DEBUG) << "processChars: mAliaseschars[" << i << "] = " << mAliaseschars[i];
    auto& id = mAliaseschars[i];
    auto& vchar = getVectorForAliasChar(id);
    LOG(DEBUG) << "vchar size = " << vchar.size();
    for (size_t j = 0; j < vchar.size(); j++) {
      LOG(DEBUG) << "DP = " << mAliaseschars[i] << " , value[" << j << "] = " << vchar[j];
    }
  }
}

//______________________________________________________________________

void DCSProcessor::processInts()
{

  // function to process aliases of Int type

  for (size_t i = 0; i != mAliasesints.size(); ++i) {
    LOG(DEBUG) << "processInts: mAliasesints[" << i << "] = " << mAliasesints[i];
    if (mIsDelta && mLatestTimestampints[i] < 0) { // we have received only the delta map, and the alias "i" was not present --> we don't process, but keep the old value in the mAvgTestInt vector
      continue;
    }
    auto& id = mAliasesints[i];
    auto& vint = getVectorForAliasInt(id);
    LOG(DEBUG) << "vint size = " << vint.size();
    for (size_t j = 0; j < vint.size(); j++) {
      LOG(DEBUG) << "DP = " << mAliasesints[i] << " , value[" << j << "] = " << vint[j];
    }
    bool isSMA = false;
    LOG(DEBUG) << "get alias = " << id.get_alias();
    // I do the moving average always of the last 2 points, no matter if it was updated or not
    doSimpleMovingAverage(2, vint, mSimpleMovingAverage[id], isSMA);
    LOG(DEBUG) << "Moving average = " << mSimpleMovingAverage[id];
    if (isSMA) {
      // create CCDB object
      mccdbSimpleMovingAverage[id.get_alias()] = mSimpleMovingAverage[id];
    }
  }
  std::map<std::string, std::string> md;
  prepareCCDBobject(mccdbSimpleMovingAverage, mccdbSimpleMovingAverageInfo, "TestDCS/IntDPs", mTF, md);
}

//______________________________________________________________________

void DCSProcessor::processDoubles()
{

  // function to process aliases of Double type
}

//______________________________________________________________________

void DCSProcessor::processUInts()
{

  // function to process aliases of UInt type
}

//______________________________________________________________________

void DCSProcessor::processBools()
{

  // function to process aliases of Bool type
}

//______________________________________________________________________

void DCSProcessor::processStrings()
{

  // function to process aliases of String type
}

//______________________________________________________________________

void DCSProcessor::processTimes()
{

  // function to process aliases of Time type
}

//______________________________________________________________________

void DCSProcessor::processBinaries()
{

  // function to process aliases of Time binary
}

//______________________________________________________________________

uint64_t DCSProcessor::processFlag(const uint64_t flags, const char* alias)
{

  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
    LOG(INFO) << "KEEP_ALIVE_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::END_FLAG) {
    LOG(INFO) << "END_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::FBI_FLAG) {
    LOG(INFO) << "FBI_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::NEW_FLAG) {
    LOG(INFO) << "NEW_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::DIRTY_FLAG) {
    LOG(INFO) << "DIRTY_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::TURN_FLAG) {
    LOG(INFO) << "TURN_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::WRITE_FLAG) {
    LOG(INFO) << "WRITE_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::READ_FLAG) {
    LOG(INFO) << "READ_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::OVERWRITE_FLAG) {
    LOG(INFO) << "OVERWRITE_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::VICTIM_FLAG) {
    LOG(INFO) << "VICTIM_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::DIM_ERROR_FLAG) {
    LOG(INFO) << "DIM_ERROR_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::BAD_DPID_FLAG) {
    LOG(INFO) << "BAD_DPID_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::BAD_FLAGS_FLAG) {
    LOG(INFO) << "BAD_FLAGS_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::BAD_TIMESTAMP_FLAG) {
    LOG(INFO) << "BAD_TIMESTAMP_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::BAD_PAYLOAD_FLAG) {
    LOG(INFO) << "BAD_PAYLOAD_FLAG active for DP " << alias;
  }
  if (flags & DataPointValue::BAD_FBI_FLAG) {
    LOG(INFO) << "BAD_FBI_FLAG active for DP " << alias;
  }

  return 0;
}

//______________________________________________________________________

void DCSProcessor::setNThreads(int n)
{

  // to set number of threads used to process the DPs

#ifdef WITH_OPENMP
  mNThreads = n > 0 ? n : 1;
#else
  LOG(WARNING) << " Multithreading is not supported, imposing single thread";
  mNThreads = 1;
#endif
}
