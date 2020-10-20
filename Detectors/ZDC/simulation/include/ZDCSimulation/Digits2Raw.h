// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <Rtypes.h>
#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
//#include "DetectorsRaw/RawFileWriter.h"
#include "DataFormatsZDC/RawEventData.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCSimulation/SimCondition.h"

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

  //  o2::raw::RawFileWriter& getWriter() { return mWriter; }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

  //  void setContinuous(bool v = true) { mIsContinuous = v; }
  bool isContinuous() const { return mIsContinuous; }
  static void print_gbt_word(UInt_t* word);

 private:
  void setTriggerMask();
  void insertLastBunch(int ibc, uint32_t orbit);
  void convertDigits(int ibc);
  void writeDigits();
  std::vector<o2::zdc::BCData> mzdcBCData, *mzdcBCDataPtr = &mzdcBCData;
  std::vector<o2::zdc::ChannelData> mzdcChData, *mzdcChDataPtr = &mzdcChData;
  EventData mZDC;                                                       /// Output structure
  bool mIsContinuous = true;                                            /// Continuous (self-triggered) or externally-triggered readout
  const ModuleConfig* mModuleConfig = 0;                                /// Trigger/readout configuration object
  const SimCondition* mSimCondition = 0;                                /// Pedestal/noise configuration object
  UShort_t mScalers[NModules][NChPerModule] = {0};                      /// ZDC orbit scalers
  UInt_t mLastOrbit = 0;                                                /// Last processed orbit
  uint32_t mTriggerMask = 0;                                            /// Trigger mask from ModuleConfig
  std::string mPrintTriggerMask = "";                                   /// Nice printout of trigger mask
  int32_t mNEmpty = -1;                                                 /// Number of clean empty bunches for pedestal evaluation
  std::array<uint16_t, o2::constants::lhc::LHCMaxBunches> mEmpty = {0}; /// Clean empty bunches along orbit
  UInt_t mLastNEmpty = 0;                                               /// Last number of empty bunches used
  Double_t mSumPed[NModules][NChPerModule] = {0};                       /// Pedestal integrated on clean empty bunches
  uint16_t mPed[NModules][NChPerModule] = {0};                          /// Current pedestal

  //  o2::raw::RawFileWriter mWriter{"ZDC"};

  int mVerbosity = 0;

  /////////////////////////////////////////////////
  ClassDefNV(Digits2Raw, 1);
};
} // namespace zdc
} // namespace o2

#endif
