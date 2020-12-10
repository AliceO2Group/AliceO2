// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  using TFType = uint64_t;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using DQDoubles = std::deque<double>;

  static constexpr int NFEACS = 8;

  TOFDCSProcessor() = default;
  ~TOFDCSProcessor() = default;

  void init(const std::vector<DPID>& pids);

  //int process(const std::vector<DPCOM>& dps);
  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  virtual uint64_t processFlags(uint64_t flag, const char* pid);

  void finalize();
  void getStripsConnectedToFEAC(int nDDL, int nFEAC, TOFFEACinfo& info) const;
  void updateFEACCCDB();
  void updateHVCCDB();

  const CcdbObjectInfo& getccdbDPsInfo() const { return mccdbDPsInfo; }
  CcdbObjectInfo& getccdbDPsInfo() { return mccdbDPsInfo; }
  const std::unordered_map<DPID, TOFDCSinfo>& getTOFDPsInfo() const { return mTOFDCS; }

  const CcdbObjectInfo& getccdbLVInfo() const { return mccdbLVInfo; }
  CcdbObjectInfo& getccdbLVInfo() { return mccdbLVInfo; }
  const std::bitset<Geo::NCHANNELS>& getLVStatus() const { return mFeac; }
  const bool isLVUpdated() const { return mUpdateFeacStatus; }

  const CcdbObjectInfo& getccdbHVInfo() const { return mccdbHVInfo; }
  CcdbObjectInfo& getccdbHVInfo() { return mccdbHVInfo; }
  const std::bitset<Geo::NCHANNELS>& getHVStatus() const { return mHV; }
  const bool isHVUpdated() const { return mUpdateHVStatus; }

  template <typename T>
  void prepareCCDBobjectInfo(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
                             const std::map<std::string, std::string>& md);

  void setTF(TFType tf) { mTF = tf; }
  void useVerboseMode() { mVerbose = true; }

 private:
  std::unordered_map<DPID, TOFDCSinfo> mTOFDCS;                // this is the object that will go to the CCDB
  std::unordered_map<DPID, bool> mPids;                        // contains all PIDs for the processor, the bool
                                                               // will be true if the DP was processed at least once
  std::unordered_map<DPID, std::vector<DPVAL>> mDpsdoublesmap; // this is the map that will hold the DPs for the
                                                               // double type (voltages and currents)

  std::array<std::array<TOFFEACinfo, NFEACS>, Geo::kNDDL> mFeacInfo;                  // contains the strip/pad info per FEAC
  std::array<std::bitset<8>, Geo::kNDDL> mPrevFEACstatus;                             // previous FEAC status
  std::bitset<Geo::NCHANNELS> mFeac;                                                  // bitset with feac status per channel
  bool mUpdateFeacStatus = false;                                                     // whether to update the FEAC status in CCDB or not
  std::bitset<Geo::NCHANNELS> mHV;                                                    // bitset with HV status per channel
  std::array<std::array<std::bitset<19>, Geo::NSECTORS>, Geo::NPLATES> mPrevHVstatus; // previous HV status
  bool mUpdateHVStatus = false;                                                       // whether to update the HV status in CCDB or not
  CcdbObjectInfo mccdbDPsInfo;
  CcdbObjectInfo mccdbLVInfo;
  CcdbObjectInfo mccdbHVInfo;
  TFType mStartTF; // TF index for processing of first processed TF, used to store CCDB object
  TFType mTF = 0;  // TF index for processing, used to store CCDB object
  bool mStartTFset = false;

  bool mVerbose = false;

  ClassDefNV(TOFDCSProcessor, 0);
};

template <typename T>
void TOFDCSProcessor::prepareCCDBobjectInfo(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
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

} // namespace tof
} // namespace o2

#endif
