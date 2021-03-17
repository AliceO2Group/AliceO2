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
//file RawReaderFT0.h class  for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FT0 data blocks and push them to DigitBlockFT0 for proccess
//TODO: prepare wrappers for containers with digits and combine classes below into one template class?
#ifndef ALICEO2_FIT_RAWREADERFT0_H_
#define ALICEO2_FIT_RAWREADERFT0_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FT0Raw/RawReaderFT0Base.h"

#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"

#include "Framework/ProcessingContext.h"
#include "Framework/DataAllocator.h"
#include "Framework/OutputSpec.h"
#include <gsl/span>

namespace o2
{
namespace ft0
{
//Normal TCM mode
template <bool useTrgInput = false>
class RawReaderFT0 : public RawReaderFT0BaseNorm
{
 public:
  RawReaderFT0(bool dumpData) : mDumpData(dumpData) {}
  RawReaderFT0(const RawReaderFT0&) = default;

  RawReaderFT0() = default;
  ~RawReaderFT0() = default;
  static constexpr bool sUseTrgInput = useTrgInput;
  void clear()
  {
    mVecDigits.clear();
    if constexpr (sUseTrgInput)
      mVecTriggerInput.clear();
    mVecChannelData.clear();
  }
  void accumulateDigits()
  {
    getDigits(mVecDigits, mVecChannelData);
    if constexpr (sUseTrgInput) {
      for (const auto& digit : mVecDigits) {
        mVecTriggerInput.emplace_back(digit.mIntRecord, digit.mTriggers.getOrA(), digit.mTriggers.getOrC(), digit.mTriggers.getVertex(), digit.mTriggers.getCen(), digit.mTriggers.getSCen());
      }
    }
    LOG(INFO) << "Number of Digits: " << mVecDigits.size();
    LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
    if constexpr (sUseTrgInput)
      LOG(INFO) << "Number of TriggerInput: " << mVecTriggerInput.size();
    if (mDumpData) {
      DigitBlockFT0::print(mVecDigits, mVecChannelData);
    }
  }
  static void prepareOutputSpec(std::vector<o2::framework::OutputSpec>& outputSpec)
  {
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe);
    if constexpr (sUseTrgInput)
      outputSpec.emplace_back(o2::header::gDataOriginFT0, "TRIGGERINPUT", 0, o2::framework::Lifetime::Timeframe);
  }
  void makeSnapshot(o2::framework::ProcessingContext& pc)
  {
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mVecDigits);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mVecChannelData);
    if constexpr (sUseTrgInput)
      pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "TRIGGERINPUT", 0, o2::framework::Lifetime::Timeframe}, mVecTriggerInput);
  }
  bool mDumpData;
  bool mUseTrgInput;
  std::vector<Digit> mVecDigits;
  std::vector<DetTrigInput> mVecTriggerInput;
  std::vector<ChannelData> mVecChannelData;
};

//Extended TCM mode (additional raw data struct)
class RawReaderFT0ext : public RawReaderFT0BaseExt
{
 public:
  RawReaderFT0ext(bool dumpData) : mDumpData(dumpData) {}
  RawReaderFT0ext(const RawReaderFT0ext&) = default;

  RawReaderFT0ext() = default;
  ~RawReaderFT0ext() = default;
  void clear()
  {
    mVecDigitsExt.clear();
    mVecChannelData.clear();
    mVecTrgExt.clear();
  }
  void accumulateDigits()
  {
    getDigits(mVecDigitsExt, mVecChannelData, mVecTrgExt);
    LOG(INFO) << "Number of Digits: " << mVecDigitsExt.size();
    LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
    LOG(INFO) << "Number of TriggerExt: " << mVecTrgExt.size();
    if (mDumpData) {
      DigitBlockFT0ext::print(mVecDigitsExt, mVecChannelData, mVecTrgExt);
    }
  }
  static void prepareOutputSpec(std::vector<o2::framework::OutputSpec>& outputSpec)
  {
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSTRGEXT", 0, o2::framework::Lifetime::Timeframe);
  }
  void makeSnapshot(o2::framework::ProcessingContext& pc)
  {
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mVecDigitsExt);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mVecChannelData);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSTRGEXT", 0, o2::framework::Lifetime::Timeframe}, mVecTrgExt);
  }
  bool mDumpData;
  std::vector<DigitExt> mVecDigitsExt;
  std::vector<ChannelData> mVecChannelData;
  std::vector<TriggersExt> mVecTrgExt;
};

} // namespace ft0
} // namespace o2

#endif