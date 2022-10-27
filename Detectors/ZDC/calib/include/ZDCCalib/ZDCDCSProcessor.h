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
#include <gsl/gsl>
#include "ZDCBase/ModuleConfig.h"
#ifndef DETECTOR_ZDCDCSPROCESSOR_H_
#define DETECTOR_ZDCDCSPROCESSOR_H_

/// @brief Class to process DCS data points

namespace o2
{
namespace zdc
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

struct ZDCDCSinfo {
  std::pair<uint64_t, double> firstValue; // first value seen by the ZDC DCS processor
  std::pair<uint64_t, double> lastValue;  // last value seen by the ZDC DCS processor
  std::pair<uint64_t, double> midValue;   // mid value seen by the ZDC DCS processor
  std::pair<uint64_t, double> maxChange;  // maximum variation seen by the ZDC DCS processor

  ZDCDCSinfo()
  {
    firstValue = std::make_pair(0, -999999999);
    lastValue = std::make_pair(0, -999999999);
    midValue = std::make_pair(0, -999999999);
    maxChange = std::make_pair(0, -999999999);
  }
  void makeEmpty()
  {
    firstValue.first = lastValue.first = midValue.first = maxChange.first = 0;
    firstValue.second = lastValue.second = midValue.second = maxChange.second = -999999999;
  }
  void print() const;

  ClassDefNV(ZDCDCSinfo, 1);
};

struct ZDCModuleMap {
  std::array<int, 4> moduleID = {-1, -1, -1, -1};
  std::array<bool, 4> readChannel = {false, false, false, false};
  std::array<bool, 4> triggerChannel = {false, false, false, false};
  std::array<int, 4> channelValue = {-1, -1, -1, -1};
};

/*struct TriggerChannelConfig {
  int id = -1;
  int first = 0;
  int last = 0;
  uint8_t shift = 0;
  int16_t threshold = 0;
};*/

class ZDCDCSProcessor
{

 public:
  using TFType = uint64_t;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

  static constexpr int NDDLS = 16;
  static constexpr int NMODULES = 8;
  static constexpr int NCHANNELS = 4;
  static constexpr int NHVCHANNELS = 22 + 12; // no. of HV+additional HV channels

  ZDCDCSProcessor() = default;
  ~ZDCDCSProcessor() = default;

  void init(const std::vector<DPID>& pids);

  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  virtual uint64_t processFlags(uint64_t flag, const char* pid);

  void getZDCActiveChannels(int nDDL, int nModule, ZDCModuleMap& info) const;
  void updateDPsCCDB();
  void updateMappingCCDB();
  void updateHVCCDB();
  void updatePositionCCDB();

  const CcdbObjectInfo& getccdbDPsInfo() const { return mccdbDPsInfo; }
  CcdbObjectInfo& getccdbDPsInfo() { return mccdbDPsInfo; }
  const std::unordered_map<DPID, ZDCDCSinfo>& getZDCDPsInfo() const { return mZDCDCS; }

  const std::bitset<NCHANNELS>& getMappingStatus() const { return mMapping; }
  bool isMappingUpdated() const { return mUpdateMapping; }

  const CcdbObjectInfo& getccdbHVInfo() const { return mccdbHVInfo; }
  CcdbObjectInfo& getccdbHVInfo() { return mccdbHVInfo; }
  const std::bitset<NHVCHANNELS>& getHVStatus() const { return mHV; }
  bool isHVUpdated() const { return mUpdateHVStatus; }

  const CcdbObjectInfo& getccdbPositionInfo() const { return mccdbPositionInfo; }
  CcdbObjectInfo& getccdbPositionInfo() { return mccdbPositionInfo; }
  const std::bitset<NCHANNELS>& getVerticalPosition() const { return mVerticalPosition; }
  bool isPositionUpdated() const { return mUpdateVerticalPosition; }

  template <typename T>
  void prepareCCDBobjectInfo(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
                             const std::map<std::string, std::string>& md);

  void setTF(TFType tf) { mTF = tf; }
  void setStartValidity(long t) { mStartValidity = t; }
  void useVerboseMode() { mVerbose = true; }

  void clearDPsinfo()
  {
    mDpsdoublesmap.clear();
    mZDCDCS.clear();
  }

 private:
  std::unordered_map<DPID, ZDCDCSinfo> mZDCDCS;                // object that will go to the CCDB
  std::unordered_map<DPID, bool> mPids;                        // contains all PIDs for the processor
                                                               // bool is true if the DP was processed at least once
  std::unordered_map<DPID, std::vector<DPVAL>> mDpsdoublesmap; // map that will hold the double value DPs

  std::array<std::array<ZDCModuleMap, NMODULES>, NDDLS> mZDCMapInfo; // contains the strip/pad info per module
  std::array<std::bitset<NCHANNELS>, NDDLS> mPreviousMapping;        // previous mapping
  std::bitset<NCHANNELS> mMapping;                                   // bitset with status per channel
  bool mUpdateMapping = false;                                       // whether to update mapping in CCDB

  std::bitset<NHVCHANNELS> mHV;           // bitset with HV status per channel
  std::bitset<NHVCHANNELS> mPrevHVstatus; // previous HV status
  bool mUpdateHVStatus = false;           // whether to update the HV status in CCDB

  std::bitset<NCHANNELS> mVerticalPosition;   // bitset with hadronic calorimeter position
  std::bitset<NCHANNELS> mPrevPositionstatus; // previous position values
  bool mUpdateVerticalPosition = false;       // whether to update the hadronic calorimeter position

  CcdbObjectInfo mccdbDPsInfo;
  CcdbObjectInfo mccdbMappingInfo;
  CcdbObjectInfo mccdbHVInfo;
  CcdbObjectInfo mccdbPositionInfo;
  TFType mStartTF; // TF index for processing of first processed TF, used to store CCDB object
  TFType mTF = 0;  // TF index for processing, used to store CCDB object
  bool mStartTFset = false;
  long mStartValidity = 0; // TF index for processing, used to store CCDB object
  bool mVerbose = false;

  ClassDefNV(ZDCDCSProcessor, 0);
};

template <typename T>
void ZDCDCSProcessor::prepareCCDBobjectInfo(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
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

} // namespace zdc
} // namespace o2

#endif
