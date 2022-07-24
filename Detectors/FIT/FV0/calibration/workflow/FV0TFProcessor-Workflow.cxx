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

#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/Digit.h"
#include "DataFormatsFV0/FV0CalibrationInfoObject.h"

using namespace o2::framework;

namespace o2::fv0
{

class FV0TFProcessor final : public o2::framework::Task
{

 public:
  void run(o2::framework::ProcessingContext& pc) final
  {
    auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
    auto channels = pc.inputs().get<gsl::span<o2::fv0::ChannelData>>("channels");
    auto digits = pc.inputs().get<gsl::span<o2::fv0::Digit>>("digits");
    auto& calib_data = pc.outputs().make<std::vector<o2::fv0::FV0CalibrationInfoObject>>(o2::framework::OutputRef{"calib", 0});
    calib_data.reserve(channels.size());

    for (const auto& channel : channels) {
      calib_data.emplace_back(channel.ChId, channel.CFDTime, channel.QTCAmpl, uint64_t(creationTime));
    }
  }
};

} // namespace o2::fv0

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"dispatcher-mode", VariantType::Bool, false, {"Dispatcher mode (FV0/SUB_DIGITSCH and FV0/SUB_DIGITSBC DPL channels should be applied as dispatcher output)."}});
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  Inputs inputs{};
  if (cfgc.options().get<bool>("dispatcher-mode")) {
    inputs.push_back(InputSpec{{"channels"}, "FV0", "SUB_DIGITSCH"});
    inputs.push_back(InputSpec{{"digits"}, "FV0", "SUB_DIGITSBC"});
  } else {
    inputs.push_back(InputSpec{{"channels"}, "FV0", "DIGITSCH"});
    inputs.push_back(InputSpec{{"digits"}, "FV0", "DIGITSBC"});
  }
  DataProcessorSpec dataProcessorSpec{
    "FV0TFProcessor",
    inputs,
    Outputs{
      {{"calib"}, "FV0", "CALIB_INFO"}},
    AlgorithmSpec{adaptFromTask<o2::fv0::FV0TFProcessor>()},
    Options{}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);
  return workflow;
}
