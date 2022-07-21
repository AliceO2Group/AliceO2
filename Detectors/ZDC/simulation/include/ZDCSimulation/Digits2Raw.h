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

/// \file Digits2Raw.h
/// \brief converts digits to raw format
/// \author pietro.cortese@cern.ch

#ifndef ALICEO2_ZDC_DIGITS2RAW_H_
#define ALICEO2_ZDC_DIGITS2RAW_H_
#include <string>
#include <bitset>
#include <Rtypes.h>
#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DataFormatsZDC/RawEventData.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCSimulation/SimCondition.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"

namespace o2
{
namespace zdc
{
class Digits2Raw
{
 public:
  Digits2Raw() = default;
  void emptyBunches(std::bitset<3564>& bunchPattern); /// Prepare list of clean empty bunches for baseline evaluation
  void processDigits(const std::string& outDir, const std::string& fileDigitsName);
  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };
  void setSimCondition(const SimCondition* SimCondition) { mSimCondition = SimCondition; };
  const SimCondition* getSimCondition() { return mSimCondition; };

  o2::raw::RawFileWriter& getWriter() { return mWriter; }

  void setFileFor(const std::string v) { mFileFor = v; }
  std::string getFileFor() const { return mFileFor; }

  void setVerbosity(int v)
  {
    mVerbosity = v;
    mWriter.setVerbosity(v);
  }
  int getVerbosity() const { return mVerbosity; }

  //  void setContinuous(bool v = true) { mIsContinuous = v; }
  bool isContinuous() const { return mIsContinuous; }
  static void print_gbt_word(const uint32_t* word, const ModuleConfig* moduleConfig = nullptr);

 private:
  void setTriggerMask();
  void updatePedestalReference(int bc);
  void resetSums(uint32_t orbit);
  void resetOutputStructure(uint16_t bc, uint32_t orbit, bool is_dummy);       /// Reset output structure not incrementing scalers for dummy bunches
  void assignTriggerBits(int ibc, uint16_t bc, uint32_t orbit, bool is_dummy); /// Assign trigger bits
  void insertLastBunch(int ibc, uint32_t orbit);                               /// Insert an empty bunch at last position in orbit
  void convertDigits(int ibc);                                                 /// Convert digits into raw data
  void writeDigits();                                                          /// Writes raw data to file
  std::vector<o2::zdc::BCData> mzdcBCData, *mzdcBCDataPtr = &mzdcBCData;
  std::vector<o2::zdc::ChannelData> mzdcChData, *mzdcChDataPtr = &mzdcChData;
  std::vector<o2::zdc::OrbitData> mzdcPedData, *mzdcPedDataPtr = &mzdcPedData;
  int mNbc = 0;
  BCData mBCD;
  EventData mZDC;                                                       /// Output structure
  bool mIsContinuous = true;                                            /// Continuous (self-triggered) or externally-triggered readout
  bool mOutputPerLink = false;                                          /// Split output
  const ModuleConfig* mModuleConfig = nullptr;                          /// Trigger/readout configuration object
  const SimCondition* mSimCondition = nullptr;                          /// Pedestal/noise configuration object
  uint16_t mScalers[NModules][NChPerModule] = {0};                      /// ZDC orbit scalers
  uint32_t mLatestOrbit = 0;                                            /// Latest processed orbit
  uint32_t mTriggerMask = 0;                                            /// Trigger mask from ModuleConfig
  std::string mPrintTriggerMask = "";                                   /// Nice printout of trigger mask
  int32_t mNEmpty = -1;                                                 /// Number of clean empty bunches for pedestal evaluation
  std::array<uint16_t, o2::constants::lhc::LHCMaxBunches> mEmpty = {0}; /// Clean empty bunches along orbit
  uint32_t mLastNEmpty = 0;                                             /// Last number of empty bunches used
  double mSumPed[NModules][NChPerModule] = {0};                         /// Pedestal integrated on clean empty bunches
  uint16_t mPed[NModules][NChPerModule] = {0};                          /// Current pedestal
  std::string mFileFor = "all";                                         /// Output file splitting
  std::string mFLP = "alio2-cr1-flp181";                                /// FLP assigned to ZDC
  o2::raw::RawFileWriter mWriter{"ZDC"};
  uint32_t mLinkID = 0;
  uint16_t mCruID = 0;
  uint16_t mFLPID = 0;
  uint32_t mEndPointID = 0;

  int mVerbosity = 0;

  /////////////////////////////////////////////////
  ClassDefNV(Digits2Raw, 1);
};
} // namespace zdc
} // namespace o2

#endif
