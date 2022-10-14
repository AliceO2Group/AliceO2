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

#include "Framework/DataProcessorSpec.h"
#include "SpacePoints/ResidualAggregator.h"
#include "TPCInterpolationWorkflow/TPCResidualAggregatorSpec.h"
#include "TPCInterpolationWorkflow/TPCUnbinnedResidualReaderSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"output-type", VariantType::String, "binnedResid", {"Comma separated list of outputs (without spaces). Valid strings: unbinnedResid, binnedResid, trackParams"}},
    {"enable-track-input", VariantType::Bool, false, {"Whether to expect track data from interpolation workflow"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", VariantType::Bool, false, {"Disables ROOT file writing"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto trkInput = configcontext.options().get<bool>("enable-track-input");

  bool writeUnbinnedResiduals = false;
  bool writeBinnedResiduals = false;
  bool writeTrackData = false;
  auto outputType = configcontext.options().get<string>("output-type");
  std::vector<std::string> outputTypes;
  size_t pos = 0;
  while ((pos = outputType.find(",")) != std::string::npos) {
    outputTypes.push_back(outputType.substr(0, pos));
    outputType.erase(0, pos + 1);
  }
  outputTypes.push_back(outputType);
  for (const auto& out : outputTypes) {
    if (out == "unbinnedResid") {
      writeUnbinnedResiduals = true;
    } else if (out == "binnedResid") {
      writeBinnedResiduals = true;
    } else if (out == "trackParams") {
      if (!trkInput) {
        LOG(error) << "Track output will be empty, because it is not configured as input";
      }
      writeTrackData = true;
    } else {
      LOG(error) << "Invalid output requested: " << out;
    }
  }

  auto fileOutput = !configcontext.options().get<bool>("disable-root-output");

  WorkflowSpec specs;
  if (!configcontext.options().get<bool>("disable-root-input")) {
    specs.emplace_back(o2::tpc::getUnbinnedTPCResidualsReaderSpec(trkInput));
  }
  specs.emplace_back(getTPCResidualAggregatorSpec(trkInput, fileOutput, writeUnbinnedResiduals, writeBinnedResiduals, writeTrackData));
  return specs;
}
