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
#include "TPCInterpolationWorkflow/TPCResidualReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"start-from-unbinned", VariantType::Bool, false, {"Do the binning of the residuals on-the-fly (taking into account allowed track-sources)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcmapcreator-workflow_configuration.ini");

  GID::mask_t allowedSources = GID::getSourcesMask("ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  GID::mask_t src = allowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));

  auto doBinning = configcontext.options().get<bool>("start-from-unbinned");

  WorkflowSpec specs;
  specs.emplace_back(o2::tpc::getTPCResidualReaderSpec(doBinning, src));
  return specs;
}
