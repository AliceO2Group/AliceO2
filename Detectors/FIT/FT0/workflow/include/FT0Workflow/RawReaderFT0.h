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
    if constexpr (sUseTrgInput) {
      mVecTriggerInput.clear();
    }
    mVecChannelData.clear();
  }
  void accumulateDigits()
  {
    if constexpr (sUseTrgInput) {
      getDigits(mVecDigits, mVecChannelData, mVecTriggerInput);
    } else {
      getDigits(mVecDigits, mVecChannelData);
    }
    LOG(INFO) << "Number of Digits: " << mVecDigits.size();
    LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
    if constexpr (sUseTrgInput) {
      LOG(INFO) << "Number of TriggerInput: " << mVecTriggerInput.size();
    }
    if (mDumpData) {
      DigitBlockFT0::print(mVecDigits, mVecChannelData);
    }
  }
  static void prepareOutputSpec(std::vector<o2::framework::OutputSpec>& outputSpec)
  {
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe);
    if constexpr (sUseTrgInput) {
      outputSpec.emplace_back(o2::header::gDataOriginFT0, "TRIGGERINPUT", 0, o2::framework::Lifetime::Timeframe);
    }
  }
  void makeSnapshot(o2::framework::ProcessingContext& pc)
  {
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mVecDigits);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mVecChannelData);
    if constexpr (sUseTrgInput) {
      pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "TRIGGERINPUT", 0, o2::framework::Lifetime::Timeframe}, mVecTriggerInput);
    }
  }
  bool mDumpData;
  std::vector<Digit> mVecDigits;
  std::vector<DetTrigInput> mVecTriggerInput;
  std::vector<ChannelData> mVecChannelData;
};

//Extended TCM mode (additional raw data struct)
template <bool useTrgInput = false>
class RawReaderFT0ext : public RawReaderFT0BaseExt
{
 public:
  RawReaderFT0ext(bool dumpData) : mDumpData(dumpData) {}
  RawReaderFT0ext(const RawReaderFT0ext&) = default;
  static constexpr bool sUseTrgInput = useTrgInput;
  RawReaderFT0ext() = default;
  ~RawReaderFT0ext() = default;
  void clear()
  {
    mVecDigits.clear();
    mVecChannelData.clear();
    mVecTrgExt.clear();
    if constexpr (sUseTrgInput) {
      mVecTriggerInput.clear();
    }
  }
  void accumulateDigits()
  {
    if constexpr (sUseTrgInput) {
      getDigits(mVecDigits, mVecChannelData, mVecTrgExt, mVecTriggerInput);
    } else {
      getDigits(mVecDigits, mVecChannelData, mVecTrgExt);
    }
    LOG(INFO) << "Number of Digits: " << mVecDigits.size();
    LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
    LOG(INFO) << "Number of TriggerExt: " << mVecTrgExt.size();
    if (mDumpData) {
      DigitBlockFT0ext::print(mVecDigits, mVecChannelData, mVecTrgExt);
    }
  }
  static void prepareOutputSpec(std::vector<o2::framework::OutputSpec>& outputSpec)
  {
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSTRGEXT", 0, o2::framework::Lifetime::Timeframe);
    if constexpr (sUseTrgInput) {
      outputSpec.emplace_back(o2::header::gDataOriginFT0, "TRIGGERINPUT", 0, o2::framework::Lifetime::Timeframe);
    }
  }
  void makeSnapshot(o2::framework::ProcessingContext& pc)
  {
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mVecDigits);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mVecChannelData);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSTRGEXT", 0, o2::framework::Lifetime::Timeframe}, mVecTrgExt);
    if constexpr (sUseTrgInput) {
      pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "TRIGGERINPUT", 0, o2::framework::Lifetime::Timeframe}, mVecTriggerInput);
    }
  }
  bool mDumpData;
  std::vector<Digit> mVecDigits;
  std::vector<ChannelData> mVecChannelData;
  std::vector<TriggersExt> mVecTrgExt;
  std::vector<DetTrigInput> mVecTriggerInput;
};

} // namespace ft0
} // namespace o2

#endif