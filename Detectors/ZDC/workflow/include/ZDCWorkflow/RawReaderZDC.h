// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawReaderZDC.h class  for RAW data reading

#ifndef ALICEO2_FIT_RAWREADERZDC_H_
#define ALICEO2_FIT_RAWREADERZDC_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>
#include "Headers/RAWDataHeader.h"
#include "DataFormatsZDC/RawEventData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/PedestalData.h"
#include "ZDCSimulation/Digits2Raw.h"
#include "ZDCSimulation/SimCondition.h"
#include "ZDCBase/ModuleConfig.h"
#include "Framework/ProcessingContext.h"
#include "Framework/DataAllocator.h"
#include "Framework/OutputSpec.h"
#include "Framework/Lifetime.h"
#include <gsl/span>

namespace o2
{
namespace zdc
{
class RawReaderZDC
{
 public:
  RawReaderZDC(bool dumpData) : mDumpData(dumpData) {}
  RawReaderZDC(const RawReaderZDC&) = default;

  RawReaderZDC() = default;
  ~RawReaderZDC() = default;

  std::map<InteractionRecord, EventData> mMapData; /// Raw data cache
  const ModuleConfig* mModuleConfig = nullptr;     /// Trigger/readout configuration object
  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };
  uint32_t mTriggerMask = 0; // Trigger mask from ModuleConfig
  void setTriggerMask();

  std::vector<o2::zdc::BCData> mDigitsBC;
  std::vector<o2::zdc::ChannelData> mDigitsCh;
  std::vector<o2::zdc::PedestalData> mPedestalData;

  void clear();

  //decoding binary data into data blocks
  void processBinaryData(gsl::span<const uint8_t> payload, int linkID); //processing data blocks into digits
  int processWord(const uint32_t* word);
  EventChData mCh; // Channel data to be decoded
  void process(const EventChData& ch);

  void accumulateDigits()
  {
    getDigits(mDigitsBC, mDigitsCh, mPedestalData);
    LOG(INFO) << "Number of Digits: " << mDigitsBC.size();
    LOG(INFO) << "Number of ChannelData: " << mDigitsCh.size();
    LOG(INFO) << "Number of PedestalData: " << mPedestalData.size();
  }

  int getDigits(std::vector<BCData>& digitsBC, std::vector<ChannelData>& digitsCh, std::vector<PedestalData>& pedestalData);
  
  static void prepareOutputSpec(std::vector<o2::framework::OutputSpec>& outputSpec)
  {
    outputSpec.emplace_back("ZDC", "DIGITSBC", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back("ZDC", "DIGITSCH", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back("ZDC", "DIGITSPD", 0, o2::framework::Lifetime::Timeframe);
  }
  void makeSnapshot(o2::framework::ProcessingContext& pc)
  {
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginZDC, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mDigitsBC);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginZDC, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mDigitsCh);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginZDC, "DIGITSPD", 0, o2::framework::Lifetime::Timeframe}, mPedestalData);
  }
  bool mDumpData;
};
} // namespace zdc
} // namespace o2

#endif
