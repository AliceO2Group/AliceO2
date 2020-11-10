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

void DCSProcessor::init(const std::vector<DPID>& pidschars, const std::vector<DPID>& pidsints,
                        const std::vector<DPID>& pidsdoubles, const std::vector<DPID>& pidsUints,
                        const std::vector<DPID>& pidsbools, const std::vector<DPID>& pidsstrings,
                        const std::vector<DPID>& pidstimes, const std::vector<DPID>& pidsbinaries)
{

  // init from separate vectors of pids (one per data point type)

  // chars
  for (auto it = std::begin(pidschars); it != std::end(pidschars); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_CHAR) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a char";
    }
    mPidschars.emplace_back((*it).get_alias(), DeliveryType::RAW_CHAR);
    mPids[*it] = mPidschars.size() - 1;
  }

  // ints
  for (auto it = std::begin(pidsints); it != std::end(pidsints); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_INT) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a int";
    }
    mPidsints.emplace_back((*it).get_alias(), DeliveryType::RAW_INT);
    mPids[*it] = mPidsints.size() - 1;
  }

  // doubles
  for (auto it = std::begin(pidsdoubles); it != std::end(pidsdoubles); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_DOUBLE) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a double";
    }
    mPidsdoubles.emplace_back((*it).get_alias(), DeliveryType::RAW_DOUBLE);
    mPids[*it] = mPidsdoubles.size() - 1;
  }

  // uints
  for (auto it = std::begin(pidsUints); it != std::end(pidsUints); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_UINT) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a uint";
    }
    mPidsUints.emplace_back((*it).get_alias(), DeliveryType::RAW_UINT);
    mPids[*it] = mPidsUints.size() - 1;
  }

  // bools
  for (auto it = std::begin(pidsbools); it != std::end(pidsbools); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_BOOL) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a bool";
    }
    mPidsbools.emplace_back((*it).get_alias(), DeliveryType::RAW_BOOL);
    mPids[*it] = mPidsbools.size() - 1;
  }

  // strings
  for (auto it = std::begin(pidsstrings); it != std::end(pidsstrings); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_STRING) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a string";
    }
    mPidsstrings.emplace_back((*it).get_alias(), DeliveryType::RAW_STRING);
    mPids[*it] = mPidsstrings.size() - 1;
  }

  // times
  for (auto it = std::begin(pidstimes); it != std::end(pidstimes); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_TIME) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a time";
    }
    mPidstimes.emplace_back((*it).get_alias(), DeliveryType::RAW_TIME);
    mPids[*it] = mPidstimes.size() - 1;
  }

  // binaries
  for (auto it = std::begin(pidsbinaries); it != std::end(pidsbinaries); ++it) {
    if ((*it).get_type() != DeliveryType::RAW_BINARY) {
      LOG(FATAL) << "Type for data point " << *it << " does not match with expectations! It should be a binary";
    }
    mPidsbinaries.emplace_back((*it).get_alias(), DeliveryType::RAW_BINARY);
    mPids[*it] = mPidsbinaries.size() - 1;
  }

  mLatestTimestampchars.resize(pidschars.size(), 0);
  mLatestTimestampints.resize(pidsints.size(), 0);
  mLatestTimestampdoubles.resize(pidsdoubles.size(), 0);
  mLatestTimestampUints.resize(pidsUints.size(), 0);
  mLatestTimestampbools.resize(pidsbools.size(), 0);
  mLatestTimestampstrings.resize(pidsstrings.size(), 0);
  mLatestTimestamptimes.resize(pidstimes.size(), 0);
  mLatestTimestampbinaries.resize(pidsbinaries.size(), 0);
}

//______________________________________________________________________

void DCSProcessor::init(const std::vector<DPID>& pids)
{

  int nchars = 0, nints = 0, ndoubles = 0, nUints = 0,
      nbools = 0, nstrings = 0, ntimes = 0, nbinaries = 0;
  for (auto it = std::begin(pids); it != std::end(pids); ++it) {
    if ((*it).get_type() == DeliveryType::RAW_CHAR) {
      mPidschars.emplace_back((*it).get_alias(), DeliveryType::RAW_CHAR);
      mPids[*it] = nchars;
      nchars++;
    }
    if ((*it).get_type() == DeliveryType::RAW_INT) {
      mPidsints.emplace_back((*it).get_alias(), DeliveryType::RAW_INT);
      mPids[*it] = nints;
      nints++;
    }
    if ((*it).get_type() == DeliveryType::RAW_DOUBLE) {
      mPidsdoubles.emplace_back((*it).get_alias(), DeliveryType::RAW_DOUBLE);
      mPids[*it] = ndoubles;
      ndoubles++;
    }
    if ((*it).get_type() == DeliveryType::RAW_UINT) {
      mPidsUints.emplace_back((*it).get_alias(), DeliveryType::RAW_UINT);
      mPids[*it] = nUints;
      nUints++;
    }
    if ((*it).get_type() == DeliveryType::RAW_BOOL) {
      mPidsbools.emplace_back((*it).get_alias(), DeliveryType::RAW_BOOL);
      mPids[*it] = nbools;
      nbools++;
    }
    if ((*it).get_type() == DeliveryType::RAW_STRING) {
      mPidsstrings.emplace_back((*it).get_alias(), DeliveryType::RAW_STRING);
      mPids[*it] = nstrings;
      nstrings++;
    }
    if ((*it).get_type() == DeliveryType::RAW_TIME) {
      mPidstimes.emplace_back((*it).get_alias(), DeliveryType::RAW_TIME);
      mPids[*it] = ntimes;
      ntimes++;
    }
    if ((*it).get_type() == DeliveryType::RAW_BINARY) {
      mPidsbinaries.emplace_back((*it).get_alias(), DeliveryType::RAW_BINARY);
      mPids[*it] = nbinaries;
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
  foundChars = processArrayType(mPidschars, DeliveryType::RAW_CHAR, map, mLatestTimestampchars, mDpscharsmap);

  // int type
  foundInts = processArrayType(mPidsints, DeliveryType::RAW_INT, map, mLatestTimestampints, mDpsintsmap);

  // double type
  foundDoubles = processArrayType(mPidsdoubles, DeliveryType::RAW_DOUBLE, map, mLatestTimestampdoubles,
                                  mDpsdoublesmap);

  // UInt type
  foundUInts = processArrayType(mPidsUints, DeliveryType::RAW_UINT, map, mLatestTimestampUints, mDpsUintsmap);

  // Bool type
  foundBools = processArrayType(mPidsbools, DeliveryType::RAW_BOOL, map, mLatestTimestampbools, mDpsboolsmap);

  // String type
  foundStrings = processArrayType(mPidsstrings, DeliveryType::RAW_STRING, map, mLatestTimestampstrings,
                                  mDpsstringsmap);

  // Time type
  foundTimes = processArrayType(mPidstimes, DeliveryType::RAW_TIME, map, mLatestTimestamptimes, mDpstimesmap);

  // Binary type
  foundBinaries = processArrayType(mPidsbinaries, DeliveryType::RAW_BINARY, map, mLatestTimestampbinaries,
                                   mDpsbinariesmap);

  if (!isDelta) {
    if (foundChars != mPidschars.size()) {
      LOG(INFO) << "Not all expected char-typed DPs found!";
    }
    if (foundInts != mPidsints.size()) {
      LOG(INFO) << "Not all expected int-typed DPs found!";
    }
    if (foundDoubles != mPidsdoubles.size()) {
      LOG(INFO) << "Not all expected double-typed DPs found!";
    }
    if (foundUInts != mPidsUints.size()) {
      LOG(INFO) << "Not all expected uint-typed DPs found!";
    }
    if (foundBools != mPidsbools.size()) {
      LOG(INFO) << "Not all expected bool-typed DPs found!";
    }
    if (foundStrings != mPidsstrings.size()) {
      LOG(INFO) << "Not all expected string-typed DPs found!";
    }
    if (foundTimes != mPidstimes.size()) {
      LOG(INFO) << "Not all expected time-typed DPs found!";
    }
    if (foundBinaries != mPidsbinaries.size()) {
      LOG(INFO) << "Not all expected binary-typed DPs found!";
    }
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
  auto el = mPids.find(dpid);
  if (el == mPids.end()) {
    LOG(ERROR) << "DP not found for this detector, please check";
    return 1;
  }
  int posInVector = (*el).second;

  if (type == DeliveryType::RAW_CHAR) {
    checkFlagsAndFill(dpcom, mLatestTimestampchars[posInVector], mDpscharsmap);
    processCharDP(dpid);
  }

  else if (type == DeliveryType::RAW_INT) {
    checkFlagsAndFill(dpcom, mLatestTimestampints[posInVector], mDpsintsmap);
    processIntDP(dpid);
  }

  else if (type == DeliveryType::RAW_DOUBLE) {
    checkFlagsAndFill(dpcom, mLatestTimestampdoubles[posInVector], mDpsdoublesmap);
    processDoubleDP(dpid);
  }

  else if (type == DeliveryType::RAW_UINT) {
    checkFlagsAndFill(dpcom, mLatestTimestampUints[posInVector], mDpsUintsmap);
    processUIntDP(dpid);
  }

  else if (type == DeliveryType::RAW_BOOL) {
    checkFlagsAndFill(dpcom, mLatestTimestampbools[posInVector], mDpsboolsmap);
    processBoolDP(dpid);
  }

  else if (type == DeliveryType::RAW_STRING) {
    checkFlagsAndFill(dpcom, mLatestTimestampstrings[posInVector], mDpsstringsmap);
    processStringDP(dpid);
  }

  else if (type == DeliveryType::RAW_TIME) {
    checkFlagsAndFill(dpcom, mLatestTimestamptimes[posInVector], mDpstimesmap);
    processTimeDP(dpid);
  }

  else if (type == DeliveryType::RAW_BINARY) {
    checkFlagsAndFill(dpcom, mLatestTimestampbinaries[posInVector], mDpsbinariesmap);
    processBinaryDP(dpid);
  }

  return 0;
}

//______________________________________________________________________

template <>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, uint64_t& latestTimeStamp,
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
    if (destmap[dpid].size() == 0 || etime != latestTimeStamp) {
      auto& tmp = destmap[dpid].emplace_back();
      std::strncpy(tmp.data(), (char*)&(val.payload_pt1), 56);
      latestTimeStamp = etime;
    }
  }
}

//______________________________________________________________________

template <>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, uint64_t& latestTimeStamp,
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
    if (destmap[dpid].size() == 0 || etime != latestTimeStamp) {
      auto& tmp = destmap[dpid].emplace_back();
      memcpy(tmp.data(), &(val.payload_pt1), 7);
      latestTimeStamp = etime;
    }
  }
}

//______________________________________________________________________

void DCSProcessor::processCharDP(const DPID& pid)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processIntDP(const DPID& pid)
{
  // processing the single pid of type int
  bool isSMA = false;
  doSimpleMovingAverage(2, mDpsintsmap[pid], mSimpleMovingAverage[pid], isSMA);
  LOG(DEBUG) << "dpid = " << pid << " --> Moving average = " << mSimpleMovingAverage[pid];
  // create CCDB object
  //if (isSMA) {
  mccdbSimpleMovingAverage[pid.get_alias()] = mSimpleMovingAverage[pid];
  //}
  return;
}

//______________________________________________________________________

void DCSProcessor::processDoubleDP(const DPID& pid)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processUIntDP(const DPID& pid)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processBoolDP(const DPID& pid)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processStringDP(const DPID& pid)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processTimeDP(const DPID& pid)
{
  // empty for the example
  return;
}

//______________________________________________________________________

void DCSProcessor::processBinaryDP(const DPID& pid)
{
  // empty for the example
  return;
}

//______________________________________________________________________

std::unordered_map<DPID, DPVAL>::const_iterator DCSProcessor::findAndCheckPid(const DPID& pid,
                                                                              DeliveryType type,
                                                                              const std::unordered_map<DPID, DPVAL>&
                                                                                map)
{

  // processing basic checks for map: all needed pids must be present
  // finds dp defined by "pid" in received map "map"

  LOG(DEBUG) << "Processing " << pid;
  auto it = map.find(pid);
  DeliveryType tt = pid.get_type();
  if (tt != type) {
    LOG(FATAL) << "Delivery Type of pid " << pid.get_alias() << " does not match definition in DCSProcessor ("
               << type << ")! Please fix";
  }
  return it;
}

//______________________________________________________________________

uint64_t DCSProcessor::processFlag(const uint64_t flags, const char* pid)
{

  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
    LOG(INFO) << "KEEP_ALIVE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::END_FLAG) {
    LOG(INFO) << "END_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::FBI_FLAG) {
    LOG(INFO) << "FBI_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::NEW_FLAG) {
    LOG(INFO) << "NEW_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIRTY_FLAG) {
    LOG(INFO) << "DIRTY_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::TURN_FLAG) {
    LOG(INFO) << "TURN_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::WRITE_FLAG) {
    LOG(INFO) << "WRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::READ_FLAG) {
    LOG(INFO) << "READ_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::OVERWRITE_FLAG) {
    LOG(INFO) << "OVERWRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::VICTIM_FLAG) {
    LOG(INFO) << "VICTIM_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIM_ERROR_FLAG) {
    LOG(INFO) << "DIM_ERROR_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_DPID_FLAG) {
    LOG(INFO) << "BAD_DPID_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FLAGS_FLAG) {
    LOG(INFO) << "BAD_FLAGS_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_TIMESTAMP_FLAG) {
    LOG(INFO) << "BAD_TIMESTAMP_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_PAYLOAD_FLAG) {
    LOG(INFO) << "BAD_PAYLOAD_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FBI_FLAG) {
    LOG(INFO) << "BAD_FBI_FLAG active for DP " << pid;
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
