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
#include "GlobalTrackingWorkflowQC/ITSTPCMatchingQCSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable use of MC information even if available"}},
    {"disable-k0-qc", o2::framework::VariantType::Bool, false, {"disable K0 QC"}},
    {"track-sources", o2::framework::VariantType::String, "ITS,TPC,ITS-TPC", {"comma-separated list of track sources to use"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  LOG(info) << "ITSTPC matching QC: disable-mc = " << configcontext.options().get<std::string>("disable-mc");
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  LOG(info) << "ITSTPC matching QC: disable-k0-qc = " << configcontext.options().get<std::string>("disable-k0-qc");
  auto doK0QC = !configcontext.options().get<bool>("disable-k0-qc");
  LOG(info) << "ITSTPC matching QC: track-sources = " << configcontext.options().get<std::string>("track-sources");
  std::string trkSources = configcontext.options().get<std::string>("track-sources");

  WorkflowSpec specs;
  specs.emplace_back(getITSTPCMatchingQCDevice(useMC, doK0QC, trkSources));
  return specs;
}
