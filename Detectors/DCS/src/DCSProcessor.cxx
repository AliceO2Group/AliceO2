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

void DCSProcessor::init(const std::unordered_map<DPID, int>& dpidmapchars,
                        const std::unordered_map<DPID, int>& dpidmapints,
                        const std::unordered_map<DPID, int>& dpidmapdoubles,
                        const std::unordered_map<DPID, int>& dpidmapUints,
                        const std::unordered_map<DPID, int>& dpidmapbools,
                        const std::unordered_map<DPID, int>& dpidmapstrings,
                        const std::unordered_map<DPID, int>& dpidmaptimes,
                        const std::unordered_map<DPID, int>& dpidmapbinaries)
{

  // init from separate vectors of aliases (one per data point type)

  // chars
  for (const auto& el : dpidmapchars) {
    if ((el.first).get_type() != DeliveryType::RAW_CHAR) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a char";
    }
    mMapchars[el.first] = el.second;
    mLatestTimestampchars[el.first] = 0;
  }

  // ints
  for (const auto& el : dpidmapints) {
    if ((el.first).get_type() != DeliveryType::RAW_INT) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a int";
    }
    mMapints[el.first] = el.second;
    mLatestTimestampints[el.first] = 0;
  }

  // doubles
  for (const auto& el : dpidmapdoubles) {
    if ((el.first).get_type() != DeliveryType::RAW_DOUBLE) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a double";
    }
    mMapdoubles[el.first] = el.second;
    mLatestTimestampdoubles[el.first] = 0;
  }

  // uints
  for (const auto& el : dpidmapUints) {
    if ((el.first).get_type() != DeliveryType::RAW_UINT) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a uint";
    }
    mMapUints[el.first] = el.second;
    mLatestTimestampUints[el.first] = 0;
  }

  // bools
  for (const auto& el : dpidmapbools) {
    if ((el.first).get_type() != DeliveryType::RAW_BOOL) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a bool";
    }
    mMapbools[el.first] = el.second;
    mLatestTimestampbools[el.first] = 0;
  }

  // strings
  for (const auto& el : dpidmapstrings) {
    if ((el.first).get_type() != DeliveryType::RAW_STRING) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a string";
    }
    mMapstrings[el.first] = el.second;
    mLatestTimestampstrings[el.first] = 0;
  }

  // times
  for (const auto& el : dpidmaptimes) {
    if ((el.first).get_type() != DeliveryType::RAW_TIME) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a time";
    }
    mMaptimes[el.first] = el.second;
    mLatestTimestamptimes[el.first] = 0;
  }

  // binaries
  for (const auto& el : dpidmapbinaries) {
    if ((el.first).get_type() != DeliveryType::RAW_BINARY) {
      LOG(FATAL) << "Type for data point " << el.first << " does not match with expectations! It should be a binary";
    }
    mMapbinaries[el.first] = el.second;
    mLatestTimestampbinaries[el.first] = 0;
  }
}

//______________________________________________________________________

void DCSProcessor::init(const std::unordered_map<DPID, int>& dpidmap)
{

  int nchars = 0, nints = 0, ndoubles = 0, nUints = 0,
      nbools = 0, nstrings = 0, ntimes = 0, nbinaries = 0;
  for (const auto& el : dpidmap) {
    if ((el.first).get_type() == DeliveryType::RAW_CHAR) {
      mMapchars[el.first] = el.second;
      mLatestTimestampchars[el.first] = 0;
      nchars++;
    }
    if ((el.first).get_type() == DeliveryType::RAW_INT) {
      mMapints[el.first] = el.second;
      mLatestTimestampints[el.first] = 0;
      nints++;
    }
    if ((el.first).get_type() == DeliveryType::RAW_DOUBLE) {
      mMapdoubles[el.first] = el.second;
      mLatestTimestampdoubles[el.first] = 0;
      ndoubles++;
    }
    if ((el.first).get_type() == DeliveryType::RAW_UINT) {
      mMapUints[el.first] = el.second;
      mLatestTimestampUints[el.first] = 0;
      nUints++;
    }
    if ((el.first).get_type() == DeliveryType::RAW_BOOL) {
      mMapbools[el.first] = el.second;
      mLatestTimestampbools[el.first] = 0;
      nbools++;
    }
    if ((el.first).get_type() == DeliveryType::RAW_STRING) {
      mMapstrings[el.first] = el.second;
      mLatestTimestampstrings[el.first] = 0;
      nstrings++;
    }
    if ((el.first).get_type() == DeliveryType::RAW_TIME) {
      mMaptimes[el.first] = el.second;
      mLatestTimestamptimes[el.first] = 0;
      ntimes++;
    }
    if ((el.first).get_type() == DeliveryType::RAW_BINARY) {
      mMapbinaries[el.first] = el.second;
      mLatestTimestampbinaries[el.first] = 0;
      nbinaries++;
    }
  }
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
  foundChars = processArrayType(mMapchars, DeliveryType::RAW_CHAR, map, mLatestTimestampchars, mDpscharsmap);

  // int type
  foundInts = processArrayType(mMapints, DeliveryType::RAW_INT, map, mLatestTimestampints, mDpsintsmap);

  // double type
  foundDoubles = processArrayType(mMapdoubles, DeliveryType::RAW_DOUBLE, map, mLatestTimestampdoubles,
                                  mDpsdoublesmap);

  // UInt type
  foundUInts = processArrayType(mMapUints, DeliveryType::RAW_UINT, map, mLatestTimestampUints, mDpsUintsmap);

  // Bool type
  foundBools = processArrayType(mMapbools, DeliveryType::RAW_BOOL, map, mLatestTimestampbools, mDpsboolsmap);

  // String type
  foundStrings = processArrayType(mMapstrings, DeliveryType::RAW_STRING, map, mLatestTimestampstrings,
                                  mDpsstringsmap);

  // Time type
  foundTimes = processArrayType(mMaptimes, DeliveryType::RAW_TIME, map, mLatestTimestamptimes, mDpstimesmap);

  // Binary type
  foundBinaries = processArrayType(mMapbinaries, DeliveryType::RAW_BINARY, map, mLatestTimestampbinaries,
                                   mDpsbinariesmap);

  if (!isDelta) {
    if (foundChars != mMapchars.size())
      LOG(INFO) << "Not all expected char-typed DPs found!";
    if (foundInts != mMapints.size())
      LOG(INFO) << "Not all expected int-typed DPs found!";
    if (foundDoubles != mMapdoubles.size())
      LOG(INFO) << "Not all expected double-typed DPs found!";
    if (foundUInts != mMapUints.size())
      LOG(INFO) << "Not all expected uint-typed DPs found!";
    if (foundBools != mMapbools.size())
      LOG(INFO) << "Not all expected bool-typed DPs found!";
    if (foundStrings != mMapstrings.size())
      LOG(INFO) << "Not all expected string-typed DPs found!";
    if (foundTimes != mMaptimes.size())
      LOG(INFO) << "Not all expected time-typed DPs found!";
    if (foundBinaries != mMapbinaries.size())
      LOG(INFO) << "Not all expected binary-typed DPs found!";
  }

  // filling CCDB info to be sent in output
  std::map<std::string, std::string> md;
  prepareCCDBobject(mccdbSimpleMovingAverage, mccdbSimpleMovingAverageInfo,
                    mName + "/TestDCS/SimpleMovingAverageDPs", mTF, md);

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
    auto el = mMapchars.find(dpid);
    if (el == mMapchars.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    auto elTime = mLatestTimestampchars.find(dpid);
    if (elTime == mLatestTimestampchars.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }
    checkFlagsAndFill(dpcom, mLatestTimestampchars[dpid], mDpscharsmap);
    processCharDP(dpid);
  }

  else if (type == DeliveryType::RAW_INT) {
    std::vector<int64_t> tmp;
    tmp.resize(100, 0);
    auto el = mMapints.find(dpid);
    if (el == mMapints.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }

    auto elTime = mLatestTimestampints.find(dpid);
    if (elTime == mLatestTimestampints.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }

    checkFlagsAndFill(dpcom, (*elTime).second, mDpsintsmap);
    //checkFlagsAndFill(dpcom, tmp[0], mDpsintsmap);
    //checkFlagsAndFill(dpcom, mLatestTimestampints[dpid], mDpsintsmap);
    processIntDP(dpid);
  }

  else if (type == DeliveryType::RAW_DOUBLE) {
    auto el = mMapdoubles.find(dpid);
    if (el == mMapdoubles.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    auto elTime = mLatestTimestampdoubles.find(dpid);
    if (elTime == mLatestTimestampdoubles.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }
    checkFlagsAndFill(dpcom, mLatestTimestampdoubles[dpid], mDpsdoublesmap);
    processDoubleDP(dpid);
  }

  else if (type == DeliveryType::RAW_UINT) {
    auto el = mMapUints.find(dpid);
    if (el == mMapUints.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    auto elTime = mLatestTimestampUints.find(dpid);
    if (elTime == mLatestTimestampUints.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }
    checkFlagsAndFill(dpcom, mLatestTimestampUints[dpid], mDpsUintsmap);
    processUIntDP(dpid);
  }

  else if (type == DeliveryType::RAW_BOOL) {
    auto el = mMapbools.find(dpid);
    if (el == mMapbools.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    auto elTime = mLatestTimestampbools.find(dpid);
    if (elTime == mLatestTimestampbools.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }
    checkFlagsAndFill(dpcom, mLatestTimestampbools[dpid], mDpsboolsmap);
    processBoolDP(dpid);
  }

  else if (type == DeliveryType::RAW_STRING) {
    auto el = mMapstrings.find(dpid);
    if (el == mMapstrings.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    auto elTime = mLatestTimestampstrings.find(dpid);
    if (elTime == mLatestTimestampstrings.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }
    checkFlagsAndFill(dpcom, mLatestTimestampstrings[dpid], mDpsstringsmap);
    processStringDP(dpid);
  }

  else if (type == DeliveryType::RAW_TIME) {
    auto el = mMaptimes.find(dpid);
    if (el == mMaptimes.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    auto elTime = mLatestTimestamptimes.find(dpid);
    if (elTime == mLatestTimestamptimes.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }
    checkFlagsAndFill(dpcom, mLatestTimestamptimes[dpid], mDpstimesmap);
    processTimeDP(dpid);
  }

  else if (type == DeliveryType::RAW_BINARY) {
    auto el = mMapbinaries.find(dpid);
    if (el == mMapbinaries.end()) {
      LOG(ERROR) << "DP not found for this detector, please check";
      return 1;
    }
    auto elTime = mLatestTimestampbinaries.find(dpid);
    if (elTime == mLatestTimestampbinaries.end()) {
      LOG(ERROR) << "Timestamp not found for this DP, please check";
      return 1;
    }
    checkFlagsAndFill(dpcom, mLatestTimestampbinaries[dpid], mDpsbinariesmap);
    processBinaryDP(dpid);
  }

  return 0;
}

//______________________________________________________________________

template <>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, int64_t& latestTimeStamp,
                                     std::unordered_map<DPID, DQStrings>& destmap)
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
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, int64_t& latestTimeStamp,
                                     std::unordered_map<DPID, DQBinaries>& destmap)
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

void DCSProcessor::processCharDP(const DPID& alias)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processIntDP(const DPID& alias)
{
  // processing the single alias of type int
  bool isSMA = false;
  doSimpleMovingAverage(2, mDpsintsmap[alias], mSimpleMovingAverage[alias], isSMA);
  LOG(DEBUG) << "dpid = " << alias << " --> Moving average = " << mSimpleMovingAverage[alias];
  // create CCDB object
  //if (isSMA) {
  mccdbSimpleMovingAverage[alias.get_alias()] = mSimpleMovingAverage[alias];
  //}
  return;
}

//______________________________________________________________________

void DCSProcessor::processDoubleDP(const DPID& alias)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processUIntDP(const DPID& alias)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processBoolDP(const DPID& alias)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processStringDP(const DPID& alias)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processTimeDP(const DPID& alias)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processBinaryDP(const DPID& alias)
{
  // empty for the example
  return;
}

//______________________________________________________________________

std::unordered_map<DPID, DPVAL>::const_iterator DCSProcessor::findAndCheckAlias(const DPID& alias,
                                                                                DeliveryType type,
                                                                                const std::unordered_map<DPID, DPVAL>&
                                                                                  map)
{

  // processing basic checks for map: all needed aliases must be present
  // finds dp defined by "alias" in received map "map"

  LOG(DEBUG) << "Processing " << alias;
  auto it = map.find(alias);
  DeliveryType tt = alias.get_type();
  if (tt != type) {
    LOG(FATAL) << "Delivery Type of alias " << alias.get_alias() << " does not match definition in DCSProcessor ("
               << type << ")! Please fix";
  }
  return it;
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
