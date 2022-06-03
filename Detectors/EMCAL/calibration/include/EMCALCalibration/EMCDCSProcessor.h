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

#ifndef ALICEO2_EMCAL_EMCDCSPROCESSOR_H_
#define ALICEO2_EMCAL_EMCDCSPROCESSOR_H_

#include <Rtypes.h>
#include <gsl/gsl>

#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"

#include "EMCALCalib/ElmbMeasurement.h"
#include "EMCALCalib/TriggerTRUDCS.h"
#include "EMCALCalib/TriggerSTUDCS.h"
#include "EMCALCalib/TriggerDCS.h"
#include "EMCALCalib/FeeDCS.h"
#include "EMCALCalib/ElmbData.h"

namespace o2
{

namespace emcal
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using EMCFEE = o2::emcal::TriggerDCS;
using EMCELMB = o2::emcal::ElmbMeasurement;

class EMCDCSProcessor
{

 public:
  using TFType = uint64_t;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

  EMCDCSProcessor() = default;
  ~EMCDCSProcessor() = default;

  void init(const std::vector<DPID>& pids);
  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  void processElmb();

  void printPDCOM(const DPCOM& dpcom);

  const FeeDCS& getFeeDCSdata() const { return *mFEECFG; }
  const ElmbData& getELMBdata() const { return *mELMBdata; }

  bool isUpdateELMB() { return mUpdateELMB; }
  bool isUpdateFEEcfg() { return mUpdateFEEcfg; }

  const CcdbObjectInfo& getccdbELMBinfo() const { return mccdbELMBinfo; }
  const CcdbObjectInfo& getccdbFeeDCSinfo() const { return mccdbFEEcfginfo; }

  CcdbObjectInfo& getccdbELMBinfo() { return mccdbELMBinfo; }
  CcdbObjectInfo& getccdbFeeDCSinfo() { return mccdbFEEcfginfo; }
  int getRunNumberFromGRP() { return mRunNumberFromGRP; }

  void updateFeeCCDBinfo();
  void updateElmbCCDBinfo();

  void useVerboseMode() { mVerbose = true; }

  template <typename T>
  void prepareCCDBobjectInfo(const T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
                             const std::map<std::string, std::string>& md);

  void setTF(TFType tf) { mTF = tf; }
  void setElmbCCDBupdateRate(TFType tf) { mElmbCCDBupdateRate = tf; }
  void setRunNumberFromGRP(int rn) { mRunNumberFromGRP = rn; }

 private:
  TFType mTF{0};                    // TF index for processing
  TFType mTFprevELMB{0};            // TF index of previous ELMB data update in CCDB
  TFType mElmbCCDBupdateRate{1000}; // duration (in TF units) for averaging and updating the ELMB data in CCDB
  bool mVerbose = false;
  int mRunNumberFromGRP = -2; // Run number from GRP; -1 is the default in RunStatusChecker.h. Here use -2.

  std::unordered_map<DPID, bool> mPids;                   // contains all PIDs for the processor, the bool
                                                          // will be true if the DP was processed at least once
  std::unordered_map<DPID, std::vector<DPVAL>> mapFEEcfg; // containds FEE CGF data

  bool mUpdateFEEcfg{false};
  bool mUpdateELMB{false};
  CcdbObjectInfo mccdbELMBinfo;
  CcdbObjectInfo mccdbFEEcfginfo;

  std::unique_ptr<FeeDCS> mFEECFG;
  std::unique_ptr<EMCELMB> mELMB;
  std::unique_ptr<ElmbData> mELMBdata;

  o2::emcal::TriggerSTUDCS mSTU;
  o2::emcal::TriggerTRUDCS mTRU;

  void FillFeeDP(const DPCOM& dpcom);
  void FillElmbDP(const DPCOM& dpcom);

  ClassDefNV(EMCDCSProcessor, 1);
};

template <typename T>
void EMCDCSProcessor::prepareCCDBobjectInfo(const T& obj, CcdbObjectInfo& info, const std::string& path, TFType tf,
                                            const std::map<std::string, std::string>& md)
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

} // namespace emcal
} // namespace o2
#endif
