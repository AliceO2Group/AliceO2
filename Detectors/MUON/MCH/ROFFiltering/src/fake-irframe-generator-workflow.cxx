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

#include "DataFormatsMCH/Digit.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "CommonDataFormat/IRFrame.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <fmt/format.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

using namespace o2::mch;
using namespace o2::framework;

/** A utility class to inject fake ITS/IRFRAMES into a workflow.
 *
 * For debug only, of course.
 *
 */
class IRFrameGenerator : public Task
{
 public:
  void init(InitContext& ic) override
  {
    std::stringstream in(ic.options().get<std::string>("irframes"));
    rapidjson::IStreamWrapper isw(in);
    rapidjson::Document d;
    d.ParseStream(isw);

    if (!d.IsArray()) {
      throw std::runtime_error(fmt::format("input string is not expected json : {}", in.str()));
    }
    for (auto& v : d.GetArray()) {
      const auto& jstart = v["min"].GetObject();
      o2::InteractionRecord start{static_cast<uint16_t>(std::stoi(jstart["bc"].GetString())),
                                  static_cast<uint32_t>(std::stoi(jstart["orbit"].GetString()))};
      const auto& jend = v["max"].GetObject();
      o2::InteractionRecord end{static_cast<uint16_t>(std::stoi(jend["bc"].GetString())),
                                static_cast<uint32_t>(std::stoi(jend["orbit"].GetString()))};
      mIRFrames.emplace_back(start, end);
    }
    LOGP(info, "Fake IRFrames to be used");
    for (const auto& ir : mIRFrames) {
      LOGP(info, "{}", ir.asString());
    }
  }

  void run(ProcessingContext& pc) override
  {
    pc.outputs().snapshot(OutputRef{"irframes"}, mIRFrames);
  }

 private:
  std::vector<o2::dataformats::IRFrame> mIRFrames;
};

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  std::string defaultFrames = R"(
[
  {
    "min": {
      "bc": "0",
      "orbit": "1021"
    },
    "max": {
      "bc": "1000",
      "orbit": "1021"
    }
  },
  {
    "min": {
      "bc": "2000",
      "orbit": "1023"
    },
    "max": {
      "bc": "2500",
      "orbit": "1023"
    }
  }
])";
  return WorkflowSpec{
    DataProcessorSpec{
      "mch-fake-irframe-generator",
      Inputs{InputSpec{"digits", "MCH", "DIGITS", 0, Lifetime::Timeframe}},
      Outputs{
        OutputSpec{{"irframes"}, "ITS", "IRFRAMES", 0, Lifetime::Timeframe}},
      AlgorithmSpec{adaptFromTask<IRFrameGenerator>()},
      Options{
        {"irframes", VariantType::String, defaultFrames, {"list of IRFrame to fake (json format)"}}}}};
}
