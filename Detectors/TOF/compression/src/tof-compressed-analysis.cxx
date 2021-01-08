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

#include "TOFWorkflowUtils/CompressedAnalysisTask.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "FairLogger.h"

using namespace o2::framework;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  auto inputDesc = ConfigParamSpec{"tof-compressed-analysis-input-desc", VariantType::String, "CRAWDATA", {"Input specs description string"}};
  workflowOptions.push_back(inputDesc);
}

#include "Framework/runDataProcessing.h" // the main driver

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto inputDesc = cfgc.options().get<std::string>("tof-compressed-analysis-input-desc");

  return WorkflowSpec{
    DataProcessorSpec{"compressed-analysis",
                      select(std::string("x:TOF/" + inputDesc).c_str()),
                      Outputs{},
                      AlgorithmSpec(adaptFromTask<o2::tof::CompressedAnalysisTask>()),
                      Options{
                        {"tof-compressed-analysis-conet-mode", VariantType::Bool, false, {"CONET mode"}},
                        {"tof-compressed-analysis-filename", VariantType::String, "", {"Analysis file name"}},
                        {"tof-compressed-analysis-function", VariantType::String, "", {"Analysis function call"}}}}};
}
