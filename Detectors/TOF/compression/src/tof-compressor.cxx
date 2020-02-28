// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tof-compressor.cxx
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  Basic DPL workflow for TOF raw data compression

#include "TOFCompression/CompressorTask.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "FairLogger.h"

using namespace o2::framework;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  auto inputDesc = ConfigParamSpec{"input-desc", VariantType::String, "RAWDATA", {"Input specs description string"}};
  auto outputDesc = ConfigParamSpec{"output-desc", VariantType::String, "CRAWDATA", {"Output specs description string"}};

  workflowOptions.push_back(inputDesc);
  workflowOptions.push_back(outputDesc);
}

#include "Framework/runDataProcessing.h" // the main driver

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  auto inputDesc = cfgc.options().get<std::string>("input-desc");
  //  auto outputDesc = cfgc.options().get<std::string>("output-desc");
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(OutputSpec(ConcreteDataTypeMatcher{"TOF", "CRAWDATA"}));

  WorkflowSpec workflow;
  workflow.emplace_back(DataProcessorSpec{
    "tof-compressor",
    select(std::string("x:TOF/" + inputDesc).c_str()),
    outputs,
    AlgorithmSpec{adaptFromTask<o2::tof::CompressorTask>()},
    Options{
      {"decoder-verbose", VariantType::Bool, false, {"Decoder verbose flag"}},
      {"encoder-verbose", VariantType::Bool, false, {"Encoder verbose flag"}},
      {"checker-verbose", VariantType::Bool, false, {"Checker verbose flag"}}}});

  return workflow;
}
