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

/// @file   FT0DataDecoderDPLSpec.h

#if defined(__has_include)
#if defined(__linux__) && (defined(__x86_64) || defined(__x86_64__)) && __has_include(<emmintrin.h>) && __has_include(<immintrin.h>) && defined(FT0_DECODER_AVX512)

#ifndef O2_FT0DATADECODERDPLSPEC_H
#define O2_FT0DATADECODERPLSPEC_H
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/SerializationMethods.h"
#include "DPLUtils/DPLRawParser.h"
#include "Framework/InputRecordWalker.h"
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <gsl/span>
#include <chrono>
#include "CommonUtils/VerbosityConfig.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/LookUpTable.h"
#include "DataFormatsFIT/Triggers.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{
class FT0DataDecoderDPLSpec : public Task
{
 public:
  FT0DataDecoderDPLSpec() = default;
  ~FT0DataDecoderDPLSpec() override = default;
  using Digit_t = o2::ft0::Digit;
  using ChannelData_t = o2::ft0::ChannelData;
  using LookupTable_t = o2::ft0::SingleLUT;
  static constexpr int sNorbits = 256;
  static constexpr int sNBC = 3564;
  static constexpr int sNlinksMax = 24;
  using NChDataBC_t = std::array<uint32_t, sNBC + 4>;
  using NChDataOrbitBC_t = std::array<NChDataBC_t, sNlinksMax>;
  std::array<std::array<uint32_t, 16>, sNlinksMax> mLUT;
  NChDataOrbitBC_t mPosChDataPerLinkOrbit[sNorbits];
  uint8_t mFEEID_TCM;
  void init(InitContext& ic) final
  {

    auto ccdbUrl = ic.options().get<std::string>("ccdb-path");
    auto lutPath = ic.options().get<std::string>("lut-path");
    mVecDigits.resize(sNorbits * sNBC);
    mVecChannelData.resize(216 * sNorbits * sNBC);
    mVecTriggers.resize(sNBC);
    // mVecChannelDataBuf.resize(216*3564);
    mVecChannelDataBuf.resize(143);
    if (ccdbUrl != "") {
      LookupTable_t::setCCDBurl(ccdbUrl);
    }
    if (lutPath != "") {
      LookupTable_t::setLUTpath(lutPath);
    }
    LookupTable_t::Instance().printFullMap();

    const auto& lut = LookupTable_t::Instance().getMapEntryPM2ChannelID();
    const auto& tcm = LookupTable_t::Instance().getEntryCRU_TCM();
    mFEEID_TCM = tcm.mLinkID + 12 * tcm.mEndPointID;
    std::array<uint32_t, 16> tmpChunk;
    std::fill_n(tmpChunk.begin(), 16, 0xff);
    std::fill_n(mLUT.begin(), 16, tmpChunk);
    for (const auto& entry : lut) {
      const auto& key = entry.first;
      const auto& value = entry.second;
      const auto feeID = key.mEntryCRU.mLinkID + 12 * key.mEntryCRU.mEndPointID;

      if (feeID >= sNlinksMax || key.mLocalChannelID >= 16) {
        LOG(warning) << "Incorrect entry: " << key.mEntryCRU.mFEEID << " " << key.mLocalChannelID;
      } else {
        mLUT[feeID][key.mLocalChannelID] = value;
      }
    }
  }
  std::vector<o2::ft0::ChannelData> mVecChannelData;
  std::vector<std::array<o2::ft0::ChannelData, 25 * 216>> mVecChannelDataBuf; // buffer per orbit
  std::vector<o2::ft0::Digit> mVecDigits;
  std::vector<o2::fit::Triggers> mVecTriggers;
  void run(ProcessingContext& pc) final;
};

framework::DataProcessorSpec getFT0DataDecoderDPLSpec(bool askSTFDist)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe);
  std::vector<InputSpec> inputSpec{{"STF", ConcreteDataTypeMatcher{"FT0", "RAWDATA"}, Lifetime::Optional}};
  if (askSTFDist) {
    inputSpec.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  std::string dataProcName = "ft0-datadecoder-dpl";
  LOG(info) << dataProcName;
  return DataProcessorSpec{
    dataProcName,
    inputSpec,
    outputSpec,
    adaptFromTask<FT0DataDecoderDPLSpec>(),
    {o2::framework::ConfigParamSpec{"ccdb-path", VariantType::String, "", {"CCDB url which contains LookupTable"}},
     o2::framework::ConfigParamSpec{"lut-path", VariantType::String, "", {"LookupTable path, e.g. FT0/LookupTable"}}}};
}

} // namespace ft0
} // namespace o2

#endif /* O2_FITDATAREADERDPL_H */
#endif
#endif
