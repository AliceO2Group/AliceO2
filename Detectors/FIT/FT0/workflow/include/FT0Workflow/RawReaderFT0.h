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

#ifndef ALICEO2_FIT_RAWREADERFT0_H_
#define ALICEO2_FIT_RAWREADERFT0_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FT0Raw/RawReaderFT0Base.h"

#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"

#include "Framework/ProcessingContext.h"

#include <gsl/span>

using namespace o2::fit;
namespace o2
{
namespace ft0
{
//Normal TCM mode
class RawReaderFT0 : public RawReaderFT0BaseNorm
{
 public:
  RawReaderFT0(bool dumpData):mDumpData(dumpData){}
  RawReaderFT0() = default;
  ~RawReaderFT0() = default;
  void clear()  {
    mVecDigits.clear();
    mVecChannelData.clear();
  }
  void print()  {
    LOG(INFO) << "Number of Digits: " << mVecDigits.size();
    LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
    if (mDumpData)  DigitBlockFT0::print(mVecDigits, mVecChannelData);
  }
  static void prepareOutputSpec(std::vector<OutputSpec> &outputSpec)
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe);
  }
  void makeSnapshot(ProcessingContext& pc)  {
    pc.outputs().snapshot(Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe}, mVecDigits);
    pc.outputs().snapshot(Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe}, mVecChannelData);
  }
  bool mDumpData;
  std::vector<Digit> mVecDigits;
  std::vector<ChannelData> mVecChannelData;
};

//Extended TCM mode (additional raw data struct)
class RawReaderFT0ext : public RawReaderFT0BaseExt
{
 public:
  RawReaderFT0ext(bool dumpData):mDumpData(dumpData){}
  RawReaderFT0ext() = default;
  ~RawReaderFT0ext() = default;
  void clear()  {
    mVecDigitsExt.clear();
    mVecChannelData.clear();
    mVecTrgExt.clear();
  }
  void print()  {
    LOG(INFO) << "Number of Digits: " << mVecDigits.size();
    LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
    LOG(INFO) << "Number of TriggerExt: " << mVecTrgExt.size();
    if (mDumpData)  DigitBlockFT0::print(mVecDigits, mVecChannelData);
  }
  static void prepareOutputSpec(std::vector<OutputSpec> &outputSpec)
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSTRGEXT", 0, Lifetime::Timeframe);
  }
  void makeSnapshot(ProcessingContext& pc)  {
    pc.outputs().snapshot(Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe}, mVecDigitsExt);
    pc.outputs().snapshot(Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe}, mVecChannelData);
    pc.outputs().snapshot(Output{o2::header::gDataOriginFT0, "DIGITSTRGEXT", 0, Lifetime::Timeframe}, mVecTrgExt);
  }
  bool mDumpData;
  std::vector<DigitExt> mVecDigitsExt;
  std::vector<ChannelData> mVecChannelData;
  std::vector<TriggersExt> mVecTrgExt;
};


} // namespace ft0
} // namespace o2

#endif