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

#ifndef DETECTOR_TOFDCSPROCESSOR_H_
#define DETECTOR_TOFDCSPROCESSOR_H_

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
#include "TOFBase/Geo.h"

/// @brief Class to process DCS data points

namespace o2
{
namespace tof
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

struct TOFDCSinfo {
  std::pair<uint64_t, double> firstValue; // first value seen by the TOF DCS processor
  std::pair<uint64_t, double> lastValue;  // last value seen by the TOF DCS processor
  std::pair<uint64_t, double> midValue;   // mid value seen by the TOF DCS processor
  std::pair<uint64_t, double> maxChange;  // maximum variation seen by the TOF DCS processor
  bool updated = false;
  TOFDCSinfo()
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

  ClassDefNV(TOFDCSinfo, 1);
};

struct TOFFEACinfo {
  std::array<int32_t, 6> stripInSM = {-1, -1, -1, -1, -1, -1};
  int32_t firstPadX = -1;
  int32_t lastPadX = -1;
};

class TOFDCSProcessor
{

 public:
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using DQDoubles = std::deque<double>;

  static constexpr int NFEACS = 8;
  static constexpr int NDDLS = Geo::kNDDL * Geo::NSECTORS;

  TOFDCSProcessor() = default;
  ~TOFDCSProcessor() = default;

  void init(const std::vector<DPID>& pids);

  //int process(const std::vector<DPCOM>& dps);
  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  uint64_t processFlags(uint64_t flag, const char* pid);

  void updateDPsCCDB();
  void getStripsConnectedToFEAC(int nDDL, int nFEAC, TOFFEACinfo& info) const;
  void updateFEACCCDB();
  void updateHVCCDB();

  const CcdbObjectInfo& getccdbDPsInfo() const { return mccdbDPsInfo; }
  CcdbObjectInfo& getccdbDPsInfo() { return mccdbDPsInfo; }
  const std::unordered_map<DPID, TOFDCSinfo>& getTOFDPsInfo() const { return mTOFDCS; }

  const CcdbObjectInfo& getccdbLVInfo() const { return mccdbLVInfo; }
  CcdbObjectInfo& getccdbLVInfo() { return mccdbLVInfo; }
  const std::bitset<Geo::NCHANNELS>& getLVStatus() const { return mFeac; }
  bool isLVUpdated() const { return mUpdateFeacStatus; }

  const CcdbObjectInfo& getccdbHVInfo() const { return mccdbHVInfo; }
  CcdbObjectInfo& getccdbHVInfo() { return mccdbHVInfo; }
  const std::bitset<Geo::NCHANNELS>& getHVStatus() const { return mHV; }
  bool isHVUpdated() const { return mUpdateHVStatus; }

  void setStartValidity(long t) { mStartValidity = t; }
  void useVerboseModeDP() { mVerboseDP = true; }
  void useVerboseModeHVLV() { mVerboseHVLV = true; }

  void clearDPsinfo()
  {
    mDpsdoublesmap.clear();
    //    mTOFDCS.clear();
  }

  bool areAllDPsFilled()
  {
    for (auto& it : mPids) {
      if (!it.second) {
        return false;
      }
    }
    return true;
  }

 private:
  std::unordered_map<DPID, TOFDCSinfo> mTOFDCS;                // this is the object that will go to the CCDB
  std::unordered_map<DPID, bool> mPids;                        // contains all PIDs for the processor, the bool
                                                               // will be true if the DP was processed at least once
  std::unordered_map<DPID, std::vector<DPVAL>> mDpsdoublesmap; // this is the map that will hold the DPs for the
                                                               // double type (voltages and currents)

  std::array<std::array<TOFFEACinfo, NFEACS>, NDDLS> mFeacInfo;                       // contains the strip/pad info per FEAC
  std::array<std::bitset<8>, NDDLS> mPrevFEACstatus;                                  // previous FEAC status
  std::bitset<Geo::NCHANNELS> mFeac;                                                  // bitset with feac status per channel
  bool mUpdateFeacStatus = false;                                                     // whether to update the FEAC status in CCDB or not
  std::bitset<Geo::NCHANNELS> mHV;                                                    // bitset with HV status per channel
  std::array<std::array<std::bitset<19>, Geo::NSECTORS>, Geo::NPLATES> mPrevHVstatus; // previous HV status
  bool mUpdateHVStatus = false;                                                       // whether to update the HV status in CCDB or not
  CcdbObjectInfo mccdbDPsInfo;
  CcdbObjectInfo mccdbLVInfo;
  CcdbObjectInfo mccdbHVInfo;
  long mFirstTime;         // time when a CCDB object was stored first
  long mStartValidity = 0; // TF index for processing, used to store CCDB object
  bool mFirstTimeSet = false;

  bool mVerboseDP = false;
  bool mVerboseHVLV = false;

  ClassDefNV(TOFDCSProcessor, 0);
};
} // namespace tof
} // namespace o2

#endif
