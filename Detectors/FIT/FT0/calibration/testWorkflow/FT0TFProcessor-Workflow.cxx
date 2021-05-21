// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
    auto channels = pc.inputs().get<gsl::span<o2::ft0::ChannelData>>("channels");
    auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
    auto& calib_data = pc.outputs().make<std::vector<o2::ft0::FT0CalibrationInfoObject>>(o2::framework::OutputRef{"calib", 0});
    calib_data.reserve(channels.size());

    for (const auto& channel : channels) {
      calib_data.emplace_back(channel.getChannelID(), channel.getTime());
    }
  }
};

} // namespace o2::ft0

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const&)
{

  DataProcessorSpec dataProcessorSpec{
    "FT0TFProcessor",
    Inputs{
      {{"channels"}, "FT0", "DIGITSCH"},
      {{"digits"}, "FT0", "DIGITSBC"},
    },
    Outputs{
      {{"calib"}, "FT0", "CALIB_INFO"}},
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0TFProcessor>()},
    Options{}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);
  return workflow;
}
