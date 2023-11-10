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

#ifndef ALICEO2_TPC_DeadChannelMapCreator_H_
#define ALICEO2_TPC_DeadChannelMapCreator_H_

#include <unordered_map>
#include <memory>

#include "Rtypes.h"

#include "CCDB/CcdbApi.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/FEEConfig.h"

namespace o2::tpc
{

enum class SourcesDeadMap : unsigned short {
  None = 0,                       ///< no inputs
  IDCPadStatus = 1 << 0,          ///< use idc pad status map
  FEEConfig = 1 << 1,             ///< use fee config
  All = IDCPadStatus | FEEConfig, ///< all sources
};
inline SourcesDeadMap operator&(SourcesDeadMap a, SourcesDeadMap b) { return static_cast<SourcesDeadMap>(static_cast<unsigned short>(a) & static_cast<unsigned short>(b)); }
inline SourcesDeadMap operator~(SourcesDeadMap a) { return static_cast<SourcesDeadMap>(~static_cast<unsigned short>(a)); }
inline SourcesDeadMap operator|(SourcesDeadMap a, SourcesDeadMap b) { return static_cast<SourcesDeadMap>(static_cast<unsigned short>(a) | static_cast<unsigned short>(b)); }

struct FEEConfig;

class DeadChannelMapCreator
{
  struct ValidityRange {
    long startvalidity = 0;
    long endvalidity = -1;
    bool isValid(long ts) const { return ts < endvalidity && ts > startvalidity; }
  };

 public:
  using CalDetFlag_t = o2::tpc::CalDet<o2::tpc::PadFlags>;

  void reset();

  void init();
  void load(long timeStampOrRun);
  void loadFEEConfigViaRunInfoTS(long timeStamp);
  void loadFEEConfigViaRunInfo(long timeStampOrRun);
  void loadFEEConfig(long tag, long createdNotAfter = -1);
  void loadIDCPadFlags(long timeStampOrRun);

  void setDeadChannelMapIDCPadStatus(const CalDetFlag_t& padStatusMap, PadFlags mask = PadFlags::flagAllNoneGood);

  const CalDet<bool>& getDeadChannelMapIDC() const { return mDeadChannelMapIDC; }
  const CalDet<bool>& getDeadChannelMapFEE() const { return mDeadChannelMapFEE; }
  const CalDet<bool>& getDeadChannelMap() const { return mDeadChannelMap; }

  void drawDeadChannelMapIDC();
  void drawDeadChannelMapFEE();
  void drawDeadChannelMap();

  long getTimeStamp(long timeStampOrRun) const;

  void finalizeDeadChannelMap();
  void resetDeadChannelMap() { mDeadChannelMap = false; }

  void setSource(SourcesDeadMap s) { mSources = s; }
  void addSource(SourcesDeadMap s) { mSources = s | mSources; }
  bool useSource(SourcesDeadMap s) const { return (mSources & s) == s; }
  SourcesDeadMap getSources() const { return mSources; }

 private:
  std::unique_ptr<FEEConfig> mFEEConfig;       ///< Electronics configuration, manually loaded
  std::unique_ptr<CalDetFlag_t> mPadStatusMap; ///< Pad status map from IDCs, manually loaded
  // FEEConfig::CalPadMapType* mPulserData;       ///< Pulser information
  // FEEConfig::CalPadMapType* mCEData;           ///< CE information

  std::unordered_map<CDBType, ValidityRange> mObjectValidity; ///< validity range of internal objects
  SourcesDeadMap mSources = SourcesDeadMap::All;              ///< Inputs to use to create the map
  ccdb::CcdbApi mCCDBApi;                                     ///< CCDB Api
  CalDet<bool> mDeadChannelMapIDC{"DeadChannelMapIDC"};       ///< Combined dead channel map
  CalDet<bool> mDeadChannelMapFEE{"DeadChannelMapFEE"};       ///< Dead Channel map from FEE configuration
  CalDet<bool> mDeadChannelMap{"DeadChannelMap"};             ///< Combined dead channel map

  ClassDefNV(DeadChannelMapCreator, 0);
};

inline long DeadChannelMapCreator::getTimeStamp(long timeStampOrRun) const
{
  if (timeStampOrRun < 1000000) {
    // assume run number
    const auto c = mCCDBApi.retrieveHeaders("RCT/Info/RunInformation", {}, timeStampOrRun);
    timeStampOrRun = (std::stol(c.at("SOR")) + std::stol(c.at("EOR"))) / 2;
  }

  return timeStampOrRun;
}

} // namespace o2::tpc

#endif
