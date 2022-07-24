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

#include "Framework/ConfigParamSpec.h"
#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/Digit.h"
#include "CommonDataFormat/FlatHisto2D.h"
using namespace o2::framework;

namespace o2::ft0
{

class FT0TimeSpectraProcessor final : public o2::framework::Task
{

 public:
  FT0TimeSpectraProcessor() = default;
  ~FT0TimeSpectraProcessor() = default;
  int mAmpThreshold{5};
  int mTimeRange{153};
  //  uint8_t mPMbitsToCheck{0b11111110};
  //  uint8_t mPMbitsGood{0b01001000};
  //  uint8_t mTrgBitsToCheck{0b11110000};
  //  uint8_t mTrgBitsGood{0b10010000};
  uint8_t mPMbitsToCheck{0};
  uint8_t mPMbitsGood{0};
  uint8_t mTrgBitsToCheck{0};
  uint8_t mTrgBitsGood{0};

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
    auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
    auto channels = pc.inputs().get<gsl::span<o2::ft0::ChannelData>>("channels");
    o2::dataformats::FlatHisto2D<float> timeSpectraInfoObject(208, 0, 208, 400, -200, 200);
    for (const auto& digit : digits) {
      if (digit.mTriggers.triggersignals & mTrgBitsToCheck != mTrgBitsGood) {
        continue;
      }
      const auto& chan = digit.getBunchChannelData(channels);
      for (const auto& channel : chan) {
        if (channel.QTCAmpl > mAmpThreshold && std::abs(channel.CFDTime) < mTimeRange && (channel.ChainQTC & mPMbitsToCheck == mPMbitsGood)) {
          const auto result = timeSpectraInfoObject.fill(channel.ChId, channel.CFDTime);
        }
      }
    }
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "CALIB_INFO", 0, o2::framework::Lifetime::Timeframe}, timeSpectraInfoObject.getBase());
  }
};

} // namespace o2::ft0

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"dispatcher-mode", VariantType::Bool, false, {"Dispatcher mode (FT0/SUB_DIGITSCH and FT0/SUB_DIGITSBC DPL channels should be applied as dispatcher output)."}});
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
  DataProcessorSpec dataProcessorSpec{
    "FT0TimeSpectraProcessor",
    inputs,
    Outputs{
      {{"calib"}, "FT0", "CALIB_INFO"}},
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0TimeSpectraProcessor>()},
    Options{}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);

  return workflow;
}
