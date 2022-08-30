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

#ifndef DETECTOR_MFTDCSPROCESSOR_H_
#define DETECTOR_MFTDCSPROCESSOR_H_

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

/// @brief Class to process DCS data points

namespace o2
{
namespace mft
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

struct MFTDCSinfo {
  std::pair<uint64_t, double> firstValue; // first value seen by the MFT DCS processor
  std::pair<uint64_t, double> lastValue;  // last value seen by the MFT DCS processor
  std::pair<uint64_t, double> midValue;   // mid value seen by the MFT DCS processor
  std::pair<uint64_t, double> maxChange;  // maximum variation seen by the MFT DCS processor
  MFTDCSinfo()
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

  ClassDefNV(MFTDCSinfo, 1);
};

class MFTDCSProcessor
{

 public:
  using TFType = uint64_t;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using DQDoubles = std::deque<double>;

  MFTDCSProcessor() = default;
  ~MFTDCSProcessor() = default;

  void init(const std::vector<DPID>& pids);

  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);

  bool sendDPsCCDB();
  void updateDPsCCDB();

  const CcdbObjectInfo& getccdbDPsInfo() const { return mccdbDPsInfo; }
  CcdbObjectInfo& getccdbDPsInfo() { return mccdbDPsInfo; }
  const std::unordered_map<DPID, MFTDCSinfo>& getMFTDPsInfo() const { return mMFTDCS; }

  template <typename T>
  void prepareCCDBobjectInfo(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf, const std::map<std::string, std::string>& md);

  void setTF(TFType tf) { mTF = tf; }
  void useVerboseMode() { mVerbose = true; }

  void setThreBackBiasCurrent(float thre)
  {
    mThresholdBackBiasCurrent = thre;
  }
  void setThreDigitCurrent(float thre)
  {
    mThresholdDigitalCurrent = thre;
  }
  void setThreAnalogCurrent(float thre)
  {
    mThresholdAnalogCurrent = thre;
  }
  void setThreBackBiasVoltage(float thre)
  {
    mThresholdBackBiasVoltage = thre;
  }
  void setThreRULV(float thre)
  {
    mThresholdRULV = thre;
  }
  void clearDPsinfo()
  {
    mDpsdoublesmap.clear();
    mMFTDCS.clear();
  }

 private:
  std::unordered_map<DPID, MFTDCSinfo> mMFTDCS;                // this is the object that will go to the CCDB
  std::unordered_map<DPID, bool> mPids;                        // contains all PIDs for the processor, the bool
                                                               // will be true if the DP was processed at least once
  std::unordered_map<DPID, std::vector<DPVAL>> mDpsdoublesmap; // this is the map that will hold the DPs for the
                                                               // double type (voltages and currents)
  CcdbObjectInfo mccdbDPsInfo;

  TFType mStartTF; // TF index for processing of first processed TF, used to store CCDB object
  TFType mTF = 0;  // TF index for processing, used to store CCDB object
  bool mStartTFset = false;

  bool mVerbose = false;

  bool mSendToCCDB = false;

  double mThresholdAnalogCurrent;
  double mThresholdBackBiasCurrent;
  double mThresholdDigitalCurrent;
  double mThresholdBackBiasVoltage;
  double mThresholdRULV;

  ClassDefNV(MFTDCSProcessor, 0);
};

template <typename T>
void MFTDCSProcessor::prepareCCDBobjectInfo(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf, const std::map<std::string, std::string>& md)
{

  // prepare all info to be sent to CCDB for object obj
  auto clName = o2::utils::MemFileHelper::getClassName(obj);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  info.setPath(path);
  info.setObjectType(clName);
  info.setFileName(flName);
  info.setStartValidityTimestamp(tf);
  info.setEndValidityTimestamp(tf + o2::ccdb::CcdbObjectInfo::MONTH);
  info.setMetaData(md);
}

} // namespace mft
} // namespace o2

#endif
