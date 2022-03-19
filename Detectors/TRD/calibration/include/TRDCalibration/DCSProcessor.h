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

#ifndef DETECTOR_TRDDCSPROCESSOR_H_
#define DETECTOR_TRDDCSPROCESSOR_H_

#include "Framework/Logger.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include <Rtypes.h>
#include <unordered_map>
#include <string>
#include <gsl/gsl>

/// @brief Class to process TRD DCS data points

namespace o2
{
namespace trd
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

struct TRDDCSMinMaxMeanInfo {
  float minValue{0.f};  // min value seen by the TRD DCS processor
  float maxValue{0.f};  // max value seen by the TRD DCS processor
  float meanValue{0.f}; // mean value seen by the TRD DCS processor
  int nPoints{0};       // number of values seen by the TRD DCS processor

  void print() const;
  void addPoint(float value);

  ClassDefNV(TRDDCSMinMaxMeanInfo, 1);
};

class DCSProcessor
{

 public:
  using TFType = uint64_t;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

  DCSProcessor() = default;
  ~DCSProcessor() = default;

  void init(const std::vector<DPID>& pids);

  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  int processFlags(uint64_t flag, const char* pid);

  void updateDPsCCDB();

  const CcdbObjectInfo& getccdbDPsInfo() const { return mCcdbDPsInfo; }
  CcdbObjectInfo& getccdbDPsInfo() { return mCcdbDPsInfo; }
  const std::unordered_map<DPID, TRDDCSMinMaxMeanInfo>& getTRDDPsInfo() const { return mTRDDCS; }

  template <typename T>
  void prepareCCDBobjectInfo(T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
                             const std::map<std::string, std::string>& md);

  void setStartValidity(TFType tf) { mStartValidity = tf; }
  void useVerboseMode() { mVerbose = true; }

  void clearDPsinfo()
  {
    mDpsDoublesmap.clear();
    mTRDDCS.clear();
  }

 private:
  std::unordered_map<DPID, TRDDCSMinMaxMeanInfo> mTRDDCS;      // this is the object that will go to the CCDB
  std::unordered_map<DPID, bool> mPids;                        // contains all PIDs for the processor, the bool
                                                               // will be true if the DP was processed at least once
  std::unordered_map<DPID, std::vector<DPCOM>> mDpsDoublesmap; // this is the map that will hold the DPs for the
                                                               // double type

  CcdbObjectInfo mCcdbDPsInfo;
  TFType mStartTF;           // TF index for processing of first processed TF, used to store CCDB object
  TFType mStartValidity = 0; // TF index for processing, used to store CCDB object
  bool mStartTFset = false;

  bool mVerbose = false;

  ClassDefNV(DCSProcessor, 0);
};

} // namespace trd
} // namespace o2

#endif
