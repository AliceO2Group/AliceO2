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
#include "MFTWorkflow/MFTAssessmentSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable use of MC information even if available"}},
    {"disable-process-gen", o2::framework::VariantType::Bool, false, {"disable processing of all generated tracks"}},
    {"finalize-analysis", o2::framework::VariantType::Bool, false, {"Process collected assessment data"}},
    {"use-full-geometry", o2::framework::VariantType::Bool, false, {"use full geometry instead of the light-weight MFT part"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2-mft-assessment.ini");
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto processGen = !configcontext.options().get<bool>("disable-process-gen");
  auto finalizeAnalysis = configcontext.options().get<bool>("finalize-analysis");
  auto useGeom = configcontext.options().get<bool>("use-full-geometry");

  LOG(info) << "MFT Assessment: disable-mc = " << configcontext.options().get<std::string>("disable-mc");
  LOG(info) << "MFT Assessment: disable-process-gen = " << configcontext.options().get<std::string>("disable-process-gen");
  LOG(info) << "MFT Assessment: finalize-analysis = " << configcontext.options().get<std::string>("finalize-analysis");

  WorkflowSpec specs;
  specs.emplace_back(o2::mft::getMFTAssessmentSpec(useMC, useGeom, processGen, finalizeAnalysis));
  return specs;
}
