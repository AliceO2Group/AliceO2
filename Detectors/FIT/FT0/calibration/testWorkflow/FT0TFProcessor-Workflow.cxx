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
#include "FT0Calibration/FT0CalibrationInfoObject.h"

using namespace o2::framework;

namespace o2::ft0
{

class FT0TFProcessor final : public o2::framework::Task
{

 public:
  void run(o2::framework::ProcessingContext& pc) final
  {
    auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
    //    LOG(info)<<" FT0TFProcessor run "<<creationTime;
    auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
    auto channels = pc.inputs().get<gsl::span<o2::ft0::ChannelData>>("channels");
    auto& calib_data = pc.outputs().make<std::vector<o2::ft0::FT0CalibrationInfoObject>>(o2::framework::OutputRef{"calib", 0});
    calib_data.reserve(channels.size());
    int nDig = digits.size();
    LOG(debug) << " nDig " << nDig;
    for (const auto& digit : digits) {
      const auto& chan = digit.getBunchChannelData(channels);
      for (const auto& channel : chan) {
        if (channel.QTCAmpl > 14 && std::abs(channel.CFDTime) < 100) {
          calib_data.emplace_back(channel.ChId, channel.CFDTime, channel.QTCAmpl, uint64_t(creationTime));
        }
      }
    }
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
    "FT0TFProcessor",
    inputs,
    Outputs{
      {{"calib"}, "FT0", "CALIB_INFO"}},
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0TFProcessor>()},
    Options{}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);

  return workflow;
}
