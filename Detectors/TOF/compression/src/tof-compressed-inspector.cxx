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

/// @file   tof-compressor.cxx
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  Basic DPL workflow for TOF raw data compression

#include "TOFWorkflowUtils/CompressedInspectorTask.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include <fairlogger/Logger.h>

using namespace o2::framework;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  auto inputDesc = ConfigParamSpec{"tof-compressed-inspector-input-desc", VariantType::String, "CRAWDATA", {"Input specs description string"}};
  auto rdhVersion = ConfigParamSpec{"tof-compressed-inspector-rdh-version", VariantType::Int, 7, {"Raw Data Header version"}};
  workflowOptions.push_back(inputDesc);
  workflowOptions.push_back(rdhVersion);
}

#include "Framework/runDataProcessing.h" // the main driver

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto inputDesc = cfgc.options().get<std::string>("tof-compressed-inspector-input-desc");
  auto rdhVersion = cfgc.options().get<int>("tof-compressed-inspector-rdh-version");

  AlgorithmSpec algoSpec;
  if (rdhVersion == 4) {
    algoSpec = AlgorithmSpec{adaptFromTask<o2::tof::CompressedInspectorTask<o2::header::RAWDataHeaderV4>>()};
  } else if (rdhVersion == 6) {
    algoSpec = AlgorithmSpec{adaptFromTask<o2::tof::CompressedInspectorTask<o2::header::RAWDataHeaderV6>>()};
  } else if (rdhVersion == 7) {
    algoSpec = AlgorithmSpec{adaptFromTask<o2::tof::CompressedInspectorTask<o2::header::RAWDataHeaderV7>>()};
  }

  WorkflowSpec workflow;
  workflow.emplace_back(DataProcessorSpec{
    "tof-compressed-inspector",
    select(std::string("x:TOF/" + inputDesc).c_str()),
    Outputs{},
    algoSpec,
    Options{
      {"tof-compressed-inspector-filename", VariantType::String, "inspector.root", {"Name of the inspector output file"}},
      {"tof-compressed-inspector-decoder-verbose", VariantType::Bool, false, {"Decode in verbose mode"}}}});

  return workflow;
}
