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

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "TRDWorkflowIO/TRDTrackReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

using namespace o2::framework;
using GTrackID = o2::dataformats::GlobalTrackID;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, true, {"disable MC propagation"}},
    {"track-types", VariantType::String, std::string{GTrackID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"output-strict", o2::framework::VariantType::Bool, false, {"outputs specs should correspond to strict matching mode"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  if (useMC) {
    LOG(warning) << "TRD track reader cannot read MC labels, useMC = false will be enforced";
    useMC = false;
  }

  GTrackID::mask_t allowedSources = GTrackID::getSourcesMask("ITS-TPC-TRD,TPC-TRD");
  GTrackID::mask_t srcTRD = allowedSources & GTrackID::getSourcesMask(configcontext.options().get<std::string>("track-types"));

  WorkflowSpec specs;
  if (GTrackID::includesSource(GTrackID::Source::ITSTPCTRD, srcTRD)) {
    specs.emplace_back(o2::trd::getTRDGlobalTrackReaderSpec(useMC));
  }
  if (GTrackID::includesSource(GTrackID::Source::TPCTRD, srcTRD)) {
    specs.emplace_back(o2::trd::getTRDTPCTrackReaderSpec(useMC, configcontext.options().get<bool>("output-strict")));
  }
  return specs;
}
