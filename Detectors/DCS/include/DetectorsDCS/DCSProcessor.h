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
#include <numeric>
#include "Framework/Logger.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"

//#ifdef WITH_OPENMP
//#include <omp.h>
//#endif

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

  using TFType = uint64_t;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

  DCSProcessor() = default;
  ~DCSProcessor() = default;

  void init(const std::vector<DPID>& pidschars, const std::vector<DPID>& pidsints,
            const std::vector<DPID>& pidsdoubles, const std::vector<DPID>& pidsUints,
            const std::vector<DPID>& pidsbools, const std::vector<DPID>& pidsstrings,
            const std::vector<DPID>& pidstimes, const std::vector<DPID>& pidsbinaries);

  void init(const std::vector<DPID>& pids);

  int processMap(const std::unordered_map<DPID, DPVAL>& map, bool isDelta = false);

  int processDP(const std::pair<DPID, DPVAL>& dpcom);

  std::unordered_map<DPID, DPVAL>::const_iterator findAndCheckPid(const DPID& pid, DeliveryType type,
                                                                  const std::unordered_map<DPID, DPVAL>& map);

  template <typename T>
  int processArrayType(const std::vector<DPID>& array, DeliveryType type, const std::unordered_map<DPID, DPVAL>& map,
                       std::vector<uint64_t>& latestTimeStamp, std::unordered_map<DPID, std::deque<T>>& destmap);

  template <typename T>
  void checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, uint64_t& latestTimeStamp,
                         std::unordered_map<DPID, T>& destmap);

  virtual void processCharDP(const DPID& pid);
  virtual void processIntDP(const DPID& pid);
  virtual void processDoubleDP(const DPID& pid);
  virtual void processUIntDP(const DPID& pid);
  virtual void processBoolDP(const DPID& pid);
  virtual void processStringDP(const DPID& pid);
  virtual void processTimeDP(const DPID& pid);
  virtual void processBinaryDP(const DPID& pid);

  virtual uint64_t processFlag(uint64_t flag, const char* pid);

  template <typename T>
  void doSimpleMovingAverage(int nelements, std::deque<T>& vect, float& avg, bool& isSMA);

  DQChars& getVectorForPidChar(const DPID& id) { return mDpscharsmap[id]; }
  DQInts& getVectorForPidInt(const DPID& id) { return mDpsintsmap[id]; }
  DQDoubles& getVectorForPidDouble(const DPID& id) { return mDpsdoublesmap[id]; }
  DQUInts& getVectorForPidUInt(const DPID& id) { return mDpsUintsmap[id]; }
  DQBools& getVectorForPidBool(const DPID& id) { return mDpsboolsmap[id]; }
  DQStrings& getVectorForPidString(const DPID& id) { return mDpsstringsmap[id]; }
  DQTimes& getVectorForPidTime(const DPID& id) { return mDpstimesmap[id]; }
  DQBinaries& getVectorForPidBinary(const DPID& id) { return mDpsbinariesmap[id]; }

  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }
  const std::unordered_map<std::string, float>& getCCDBSimpleMovingAverage() const { return mccdbSimpleMovingAverage; }
  const CcdbObjectInfo& getCCDBSimpleMovingAverageInfo() const { return mccdbSimpleMovingAverageInfo; }
  CcdbObjectInfo& getCCDBSimpleMovingAverageInfo() { return mccdbSimpleMovingAverageInfo; }

  void setTF(TFType tf) { mTF = tf; }

  void setIsDelta(bool isDelta) { mIsDelta = isDelta; }
  void isDelta() { mIsDelta = true; }
  bool getIsDelta() const { return mIsDelta; }

  void setMaxCyclesNoFullMap(uint64_t maxCycles) { mMaxCyclesNoFullMap = maxCycles; }
  uint64_t getMaxCyclesNoFullMap() const { return mMaxCyclesNoFullMap; }

  uint64_t getNCyclesNoFullMap() const { return mNCyclesNoFullMap; }

  template <typename T>
  void prepareCCDBobject(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
                         const std::map<std::string, std::string>& md);

  void setName(std::string name) { mName = name; }
  const std::string getName() const { return mName; }

 private:
  bool mFullMapSent = false;                            // set to true as soon as a full map was sent. No delta can
                                                        // be received if there was never a full map sent
  int64_t mNCyclesNoFullMap = 0;                        // number of times the delta was sent withoug a full map
  int64_t mMaxCyclesNoFullMap = 6000;                   // max number of times when the delta can be sent between
                                                        // two full maps (assuming DCS sends data every 50 ms, this
                                                        // means a 5 minutes threshold)
  bool mIsDelta = false;                                // set to true in case you are processing  delta map
                                                        // (containing only DPs that changed)
  std::unordered_map<DPID, float> mSimpleMovingAverage; // moving average for several DPs
  std::unordered_map<DPID, DQChars> mDpscharsmap;
  std::unordered_map<DPID, DQInts> mDpsintsmap;
  std::unordered_map<DPID, DQDoubles> mDpsdoublesmap;
  std::unordered_map<DPID, DQUInts> mDpsUintsmap;
  std::unordered_map<DPID, DQBools> mDpsboolsmap;
  std::unordered_map<DPID, DQStrings> mDpsstringsmap;
  std::unordered_map<DPID, DQTimes> mDpstimesmap;
  std::unordered_map<DPID, DQBinaries> mDpsbinariesmap;
  std::vector<DPID> mPidschars;
  std::vector<DPID> mPidsints;
  std::vector<DPID> mPidsdoubles;
  std::vector<DPID> mPidsUints;
  std::vector<DPID> mPidsbools;
  std::vector<DPID> mPidsstrings;
  std::vector<DPID> mPidstimes;
  std::vector<DPID> mPidsbinaries;
  std::unordered_map<DPID, int> mPids; // contains all PIDs for the current processor; the value correspond to the index
                                       // in the mPidschars/ints/doubles.. vectors
  std::vector<uint64_t> mLatestTimestampchars;
  std::vector<uint64_t> mLatestTimestampints;
  std::vector<uint64_t> mLatestTimestampdoubles;
  std::vector<uint64_t> mLatestTimestampUints;
  std::vector<uint64_t> mLatestTimestampbools;
  std::vector<uint64_t> mLatestTimestampstrings;
  std::vector<uint64_t> mLatestTimestamptimes;
  std::vector<uint64_t> mLatestTimestampbinaries;
  int mNThreads = 1;                                               // number of  threads
  std::unordered_map<std::string, float> mccdbSimpleMovingAverage; // unordered map in which to store the CCDB entry
                                                                   // for the DPs for which we calculated the simple
                                                                   // moving average
  CcdbObjectInfo mccdbSimpleMovingAverageInfo;                     // info to store the output of the calibration for
                                                                   // the DPs for which we calculated
                                                                   // the simple moving average
  TFType mTF = 0;                                                  // TF index for processing,
                                                                   // used to store CCDB object
  std::string mName = "";                                          // to be used to determine CCDB path

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
int DCSProcessor::processArrayType(const std::vector<DPID>& array,
                                   DeliveryType type, const std::unordered_map<DPID, DPVAL>& map,
                                   std::vector<uint64_t>& latestTimeStamp,
                                   std::unordered_map<DPID, std::deque<T>>& destmap)
{

  // processing the array of type T

  int found = 0;
  //#ifdef WITH_OPENMP
  //omp_set_num_threads(mNThreads);
  //#pragma omp parallel for schedule(dynamic)
  //#endif
  for (size_t i = 0; i != array.size(); ++i) {
    auto it = findAndCheckPid(array[i], type, map);
    if (it == map.end()) {
      if (!mIsDelta) {
        LOG(ERROR) << "Element " << array[i] << " not found " << std::endl;
      }
      continue;
    }
    found++;
    std::pair<DPID, DPVAL> pairIt = *it;
    checkFlagsAndFill(pairIt, latestTimeStamp[i], destmap);
    if (type == RAW_CHAR) {
      processCharDP(array[i]);
    } else if (type == RAW_INT) {
      processIntDP(array[i]);
    } else if (type == RAW_DOUBLE) {
      processDoubleDP(array[i]);
    } else if (type == RAW_UINT) {
      processUIntDP(array[i]);
    } else if (type == RAW_BOOL) {
      processBoolDP(array[i]);
    } else if (type == RAW_STRING) {
      processStringDP(array[i]);
    } else if (type == RAW_TIME) {
      processTimeDP(array[i]);
    } else if (type == RAW_BINARY) {
      processBinaryDP(array[i]);
    }
    // todo: better to move the "found++" after the process, in case it fails?
  }
  return found;
}

//______________________________________________________________________

template <typename T>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, uint64_t& latestTimeStamp,
                                     std::unordered_map<DPID, T>& destmap)
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
      LOG(DEBUG) << "adding new value";
      destmap[dpid].push_back(val.payload_pt1);
      latestTimeStamp = etime;
    }
  }
}

//______________________________________________________________________

template <>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, uint64_t& latestTimeStamp,
                                     std::unordered_map<DPID, DCSProcessor::DQStrings>& destmap);

//______________________________________________________________________

template <>
void DCSProcessor::checkFlagsAndFill(const std::pair<DPID, DPVAL>& dpcom, uint64_t& latestTimeStamp,
                                     std::unordered_map<DPID, DCSProcessor::DQBinaries>& destmap);

//______________________________________________________________________

template <typename T>
void DCSProcessor::doSimpleMovingAverage(int nelements, std::deque<T>& vect, float& avg, bool& isSMA)
{

  // Do simple moving average on vector of type T

  if (vect.size() <= nelements) {
    //avg = std::accumulate(vect.begin(), vect.end(), 0.0) / vect.size();
    avg = (avg * (vect.size() - 1) + vect.back()) / vect.size();
    return;
  }

  avg += (vect[vect.size() - 1] - vect[0]) / nelements;
  vect.pop_front();
  isSMA = true;
}

//______________________________________________________________________

template <typename T>
void DCSProcessor::prepareCCDBobject(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
                                     const std::map<std::string, std::string>& md)
{

  // prepare all info to be sent to CCDB for object obj
  auto clName = o2::utils::MemFileHelper::getClassName(obj);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  info.setPath(path);
  info.setObjectType(clName);
  info.setFileName(flName);
  info.setStartValidityTimestamp(tf);
  info.setEndValidityTimestamp(99999999999999);
  info.setMetaData(md);
}
} // namespace dcs
} // namespace o2

#endif
