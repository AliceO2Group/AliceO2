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
#include "Framework/CompletionPolicyHelpers.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "TRDWorkflowIO/TRDCalibWriterSpec.h"
#include "TRDWorkflowIO/TRDTrackWriterSpec.h"
#include "TRDWorkflow/TrackBasedCalibSpec.h"
#include "TRDWorkflow/TRDGlobalTrackingSpec.h"
#include "TRDWorkflow/TRDGlobalTrackingQCSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

using namespace o2::framework;
using GTrackID = o2::dataformats::GlobalTrackID;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRD|trd).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"Disable MC labels"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}},
    {"enable-trackbased-calib", o2::framework::VariantType::Bool, false, {"enable calibration devices which are based on tracking output"}},
    {"enable-qc", o2::framework::VariantType::Bool, false, {"enable tracking QC"}},
    {"track-sources", VariantType::String, std::string{GTrackID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"filter-trigrec", o2::framework::VariantType::Bool, false, {"ignore interaction records without ITS data"}},
    {"strict-matching", o2::framework::VariantType::Bool, false, {"High purity preliminary matching"}},
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
  auto strict = configcontext.options().get<bool>("strict-matching");
  GTrackID::mask_t srcTRD = allowedSources & GTrackID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  if (strict && (srcTRD & ~GTrackID::getSourcesMask("TPC")).any()) {
    LOGP(warning, "In strict matching mode only TPC source allowed, {} asked, redefining", GTrackID::getSourcesNames(srcTRD));
    srcTRD = GTrackID::getSourcesMask("TPC");
  }
  o2::framework::WorkflowSpec specs;
  bool useMC = !configcontext.options().get<bool>("disable-mc");

  // processing devices
  specs.emplace_back(o2::trd::getTRDGlobalTrackingSpec(useMC, srcTRD, trigRecFilterActive, strict));
  if (configcontext.options().get<bool>("enable-trackbased-calib")) {
    specs.emplace_back(o2::trd::getTRDTrackBasedCalibSpec(srcTRD));
  }
  if (configcontext.options().get<bool>("enable-qc")) {
    specs.emplace_back(o2::trd::getTRDGlobalTrackingQCSpec(srcTRD));
  }

  // output devices
  if (!configcontext.options().get<bool>("disable-root-output")) {
    if (GTrackID::includesSource(GTrackID::Source::ITSTPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDGlobalTrackWriterSpec(useMC));
    }
    if (GTrackID::includesSource(GTrackID::Source::TPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDTPCTrackWriterSpec(useMC, strict));
    }
    if (configcontext.options().get<bool>("enable-trackbased-calib")) {
      specs.emplace_back(o2::trd::getTRDCalibWriterSpec());
    }
  }

  // input
  auto maskClusters = GTrackID::getSourcesMask("TRD,TPC");
  auto maskTracks = srcTRD | GTrackID::getSourcesMask("TPC"); // we always need the TPC tracks for the refit
  if (GTrackID::includesDet(GTrackID::DetID::ITS, srcTRD)) {
    maskClusters |= GTrackID::getSourcesMask("ITS");
    maskTracks |= GTrackID::getSourcesMask("ITS");
  }
  auto maskMatches = GTrackID::getSourcesMask(GTrackID::NONE);
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, maskClusters, maskMatches, maskTracks, useMC);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return specs;
}
