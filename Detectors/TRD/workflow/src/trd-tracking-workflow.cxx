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
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "TRDWorkflowIO/TRDCalibWriterSpec.h"
#include "TRDWorkflowIO/TRDTrackWriterSpec.h"
#include "TRDWorkflow/TrackBasedCalibSpec.h"
#include "TRDWorkflow/TRDGlobalTrackingSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

using namespace o2::framework;
using GTrackID = o2::dataformats::GlobalTrackID;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"Disable MC labels"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}},
    {"tracking-sources", VariantType::String, std::string{GTrackID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"filter-trigrec", o2::framework::VariantType::Bool, false, {"ignore interaction records without ITS data"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  GTrackID::mask_t allowedSources = GTrackID::getSourcesMask("ITS-TPC,TPC");
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2trdtracking-workflow_configuration.ini");
  auto trigRecFilterActive = configcontext.options().get<bool>("filter-trigrec");
  GTrackID::mask_t srcTRD = allowedSources & GTrackID::getSourcesMask(configcontext.options().get<std::string>("tracking-sources"));

  o2::framework::WorkflowSpec specs;
  bool useMC = false;
  if (!configcontext.options().get<bool>("disable-mc") && !useMC) {
    LOG(WARNING) << "MC is not disabled, although it is not yet supported by the workflow. It is forced off.";
  }

  // processing devices
  specs.emplace_back(o2::trd::getTRDGlobalTrackingSpec(useMC, srcTRD, trigRecFilterActive));
  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, srcTRD)) {
    specs.emplace_back(o2::trd::getTRDTrackBasedCalibSpec());
  }

  // output devices
  if (!configcontext.options().get<bool>("disable-root-output")) {
    if (GTrackID::includesSource(GTrackID::Source::ITSTPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDGlobalTrackWriterSpec(useMC));
      specs.emplace_back(o2::trd::getTRDCalibWriterSpec());
    }
    if (GTrackID::includesSource(GTrackID::Source::TPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDTPCTrackWriterSpec(useMC));
    }
  }

  // input
  auto maskClusters = GTrackID::getSourcesMask("TRD");
  auto maskTracks = srcTRD & GTrackID::getSourcesMask("TPC");
  auto maskMatches = srcTRD & GTrackID::getSourcesMask("ITS-TPC");
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, maskClusters, maskMatches, maskTracks, useMC);

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
