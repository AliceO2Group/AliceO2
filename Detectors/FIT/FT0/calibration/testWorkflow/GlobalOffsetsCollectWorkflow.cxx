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
#include "DataFormatsFT0/GlobalOffsetsInfoObject.h"
#include "DataFormatsFT0/RecPoints.h"
#include "Framework/Logger.h"
using namespace o2::framework;

namespace o2::ft0
{

class GlobalOffsetsCollectWorkflow final : public o2::framework::Task
{

 public:
  void run(o2::framework::ProcessingContext& pc) final
  {
    auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
    auto recpoints = pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("recpoints");
    auto& calib_data = pc.outputs().make<std::vector<o2::ft0::GlobalOffsetsInfoObject>>(o2::framework::OutputRef{"calib", 0});
    for (const auto& recpoint : recpoints) {
      short t0AC = recpoint.getCollisionTimeMean();
      if (std::abs(t0AC) < 1000) {
        calib_data.emplace_back(t0AC, uint64_t(creationTime));
      }
    }
  }
};

} // namespace o2::ft0

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  Inputs inputs{};
  inputs.push_back(InputSpec{{"recpoints"}, "FT0", "RECPOINTS"});

  DataProcessorSpec dataProcessorSpec{
    "collect-global-offsets",
    inputs,
    Outputs{
      {{"calib"}, "FT0", "CALIB_INFO"}},
    AlgorithmSpec{adaptFromTask<o2::ft0::GlobalOffsetsCollectWorkflow>()},
    Options{}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);

  return workflow;
}
