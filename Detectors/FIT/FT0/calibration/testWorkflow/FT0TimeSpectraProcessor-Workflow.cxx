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

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"
#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "FT0Base/Geometry.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/DigitFilterParam.h"
#include "CommonDataFormat/FlatHisto2D.h"

using namespace o2::framework;

namespace o2::ft0
{

class FT0TimeSpectraProcessor final : public o2::framework::Task
{

 public:
  static constexpr int sNCHANNELS = Geometry::Nchannels;
  static constexpr int sNCHANNELS_A = 4 * Geometry::NCellsA;

  int mNbinsY{400};
  float mMinY{-200.};
  float mMaxY{200.};
  int mAmpThreshold{10};
  int mTimeWindow{153};
  uint8_t mPMbitsToCheck{0b11111110};
  uint8_t mPMbitsGood{0b01001000};
  uint8_t mTrgBitsToCheck{0b11110000};
  uint8_t mTrgBitsGood{0b10010000};
  void init(o2::framework::InitContext& ic) final
  {
    mNbinsY = ic.options().get<int>("number-bins");
    mMinY = ic.options().get<float>("low-edge");
    mMaxY = ic.options().get<float>("upper-edge");
    LOG(info) << "Histogram parameters: " << mNbinsY << " " << mMinY << " " << mMaxY;
    const auto& param = o2::ft0::DigitFilterParam::Instance();
    param.printKeyValues();
    mAmpThreshold = param.mAmpThreshold;
    mTimeWindow = param.mTimeWindow;
    mPMbitsGood = param.mPMbitsGood;
    mPMbitsToCheck = param.mPMbitsToCheck;
    mTrgBitsGood = param.mTrgBitsGood;
    mTrgBitsToCheck = param.mTrgBitsToCheck;
  }
  void run(o2::framework::ProcessingContext& pc) final
  {
    enum { kTimeA = sNCHANNELS,
           kTimeC = sNCHANNELS + 1,
           kSumTimeAC = sNCHANNELS + 2,
           kDiffTimeCA = sNCHANNELS + 3
    };
    const int nExtraSpectraSlots = 4; // For timeA/C sum and diff
    auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
    auto channels = pc.inputs().get<gsl::span<o2::ft0::ChannelData>>("channels");
    o2::dataformats::FlatHisto2D<float> timeSpectraInfoObject(sNCHANNELS + nExtraSpectraSlots, 0, sNCHANNELS + nExtraSpectraSlots, mNbinsY, mMinY, mMaxY);
    for (const auto& digit : digits) {
      if ((digit.mTriggers.getTriggersignals() & mTrgBitsToCheck) != mTrgBitsGood) {
        continue;
      }
      const auto& chan = digit.getBunchChannelData(channels);
      float timeA{0}, timeC{0};
      int NchanA{0}, NchanC{0};
      for (const auto& channel : chan) {
        if (channel.QTCAmpl > mAmpThreshold && std::abs(channel.CFDTime) < mTimeWindow && ((channel.ChainQTC & mPMbitsToCheck) == mPMbitsGood && channel.ChId < sNCHANNELS)) {
          const auto result = timeSpectraInfoObject.fill(channel.ChId, channel.CFDTime);
          if (channel.ChId < sNCHANNELS_A) {
            NchanA++;
            timeA += channel.CFDTime;
          } else {
            NchanC++;
            timeC += channel.CFDTime;
          }
        }
      }
      if (NchanA > 0) {
        timeA = timeA / NchanA;
        const auto result = timeSpectraInfoObject.fill(kTimeA, timeA);
      }
      if (NchanC > 0) {
        timeC = timeC / NchanC;
        const auto result = timeSpectraInfoObject.fill(kTimeC, timeC);
        if (NchanA > 0) {
          const auto resultSum = timeSpectraInfoObject.fill(kSumTimeAC, (timeA + timeC) / 2);
          const auto resultDiff = timeSpectraInfoObject.fill(kDiffTimeCA, (timeC - timeA) / 2.);
        }
      }
    }

    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "TIME_SPECTRA", 0}, timeSpectraInfoObject.getBase());
  }
};

} // namespace o2::ft0

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"dispatcher-mode", VariantType::Bool, false, {"Dispatcher mode (FT0/SUB_DIGITSCH and FT0/SUB_DIGITSBC DPL channels should be applied as dispatcher output)."}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  Inputs inputs{};
  if (cfgc.options().get<bool>("dispatcher-mode")) {
    inputs.push_back(InputSpec{{"channels"}, "FT0", "SUB_DIGITSCH"});
    inputs.push_back(InputSpec{{"digits"}, "FT0", "SUB_DIGITSBC"});
  } else {
    inputs.push_back(InputSpec{{"channels"}, "FT0", "DIGITSCH"});
    inputs.push_back(InputSpec{{"digits"}, "FT0", "DIGITSBC"});
  }
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  DataProcessorSpec dataProcessorSpec{
    "FT0TimeSpectraProcessor",
    inputs,
    Outputs{
      {{"timeSpectra"}, "FT0", "TIME_SPECTRA"}},
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0TimeSpectraProcessor>()},
    Options{{"number-bins", VariantType::Int, 400, {"Number of bins along Y-axis"}},
            {"low-edge", VariantType::Float, -200.0f, {"Lower edge of first bin along Y-axis"}},
            {"upper-edge", VariantType::Float, 200.0f, {"Upper edge of last bin along Y-axis"}}}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);

  return workflow;
}
