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

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/DataProcessorSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "GlobalTrackingWorkflow/GlobalFwdMatchingSpec.h"
#include "GlobalTrackingWorkflow/GlobalFwdTrackWriterSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2matchmftmch-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::globaltracking::getGlobalFwdMatchingSpec(useMC));
  if (!configcontext.options().get<bool>("disable-root-output")) {
    specs.emplace_back(o2::globaltracking::getGlobalFwdTrackWriterSpec(useMC));
  }

  auto srcTracks = GID::getSourcesMask("MFT,MCH");
  auto srcClusters = GID::getSourcesMask("MFT");

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcClusters, GID::getSourcesMask(""), srcTracks, useMC, srcClusters, srcTracks);

  return std::move(specs);
}
