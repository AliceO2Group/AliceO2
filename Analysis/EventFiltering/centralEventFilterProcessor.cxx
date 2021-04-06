// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   centralEventFilterProcessor.cxx

#include "centralEventFilterProcessor.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

#include <cstdio>
#include <fmt/format.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

using namespace o2::framework;
using namespace rapidjson;

namespace {
  Document readJsonFile(std::string& config) {
    FILE* fp = fopen(config.data(), "rb");

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    Document d;
    d.ParseStream(is);
    fclose(fp);
    return d;
  }
}

namespace o2::aod::filtering
{

void CentralEventFilterProcessor::init(framework::InitContext& ic)
{
  Document d = readJsonFile(config);
  const Value& workflows = d["workflows"];
    // JSON example
    // {
    //   "subwagon_name" : "CentralEventFilterProcessor",
    //   "configuration" : {
    //     "NucleiFilters" : {
    //       "H2" : 0.1,
    //       "H3" : 0.3,
    //       "HE3" : 1.,
    //       "HE4" : 1.
    //     }
    //   }
    // }
  for (auto& workflow : workflows) {
    if (std::string_view(workflow["subwagon_name"]) == "CentralEventFilterProcessor") {
      auto& config = workflow["configuration"]);
      for (auto& filter : AvailableFilters) {
        auto& filterConfig = config[filter];
        for (auto& node : filterConfig) {
          mDownscaling[node.name] = node.value;
        }
      }
      break;
    }
  }
}

void CentralEventFilterProcessor::run(ProcessingContext& pc)
{
 
}

DataProcessorSpec getCentralEventFilterProcessorSpec(std::string& config)
{

  std::vector<InputSpec> inputs;
  for (auto& workflow : workflows) {
    for (unsigned int iFilter{0}; iFilter < AvailableFilters.size(); ++iFilter) {
      if (std::string_view(workflow["subwagon_name"]) == std::string_view(AvailableFilters[iFilter])) {
        inputs.emplace_back(AvailableFilters[iFilter], "AOD", FilterDescriptions[iFilter], 0, Lifetime::Timeframe);
        break;
      }
    }
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("AOD", "Decision", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "o2-central-event-filter-processor",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CentralEventFilterProcessor>(config)},
    Options{
      {"filtering-config", VariantType::String, "", {"Path to the filtering json config file"}}}};
}

} // namespace o2::aod::filtering