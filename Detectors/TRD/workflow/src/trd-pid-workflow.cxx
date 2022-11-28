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

/// \file trd-pid-workflow.cxx
/// \brief This file defines the workflow for the pid signal.
/// \author Felix Schlepper

#include "TRDWorkflow/TRDPIDSpec.h"
#include "TRDWorkflowIO/TRDPIDWriterSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "DataFormatsTRD/PID.h"
#include "Framework/Logger.h"

using namespace o2::framework;
using GTrackID = o2::dataformats::GlobalTrackID;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRD|trd).*[W,w]riter.*"));
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", VariantType::Bool, false, {"Disable MC labels"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GTrackID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"policy", VariantType::String, std::string{"default"}, {"Policy for PID evaluation"}},
  };
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2trdpid-workflow_configuration.ini");
  GTrackID::mask_t allowedSources = GTrackID::getSourcesMask("ITS-TPC-TRD,TPC-TRD");
  GTrackID::mask_t srcTRD = allowedSources & GTrackID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));

  // Parse policy string
  auto policyStr = configcontext.options().get<std::string>("policy");
  auto policyIt = o2::trd::PIDPolicyString.find(policyStr);
  if (policyIt == o2::trd::PIDPolicyString.end()) {
    throw std::runtime_error(fmt::format("No PID model named {:s} available!", policyStr));
  }
  o2::trd::PIDPolicy policy = policyIt->second;
  LOGF(info, "Using PID policy %s(%u)", policyStr, static_cast<unsigned int>(policy));

  // MC labels are passed through for the global tracking downstream
  // in case ROOT output is requested the tracklet labels are duplicated
  bool useMC = !configcontext.options().get<bool>("disable-mc");

  WorkflowSpec specs;
  specs.emplace_back(o2::trd::getTRDPIDSpec(useMC, policy, srcTRD));

  if (!configcontext.options().get<bool>("disable-root-output")) {
    if (GTrackID::includesSource(GTrackID::Source::ITSTPCTRD, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDPIDGlobalWriterSpec(useMC));
    }
    if (GTrackID::includesSource(GTrackID::Source::TPCTRD, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDPIDTPCWriterSpec(useMC));
    }
  }

  // input
  auto maskClusters = GTrackID::getSourcesMask("TRD");
  auto maskTracks = srcTRD | GTrackID::getSourcesMask("TPC-TRD"); // minimum TRD track
  if (GTrackID::includesDet(GTrackID::DetID::ITS, srcTRD)) {
    maskTracks |= GTrackID::getSourcesMask("ITS-TPC-TRD");
  }
  auto maskMatches = GTrackID::getSourcesMask(GTrackID::NONE);
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, maskClusters, maskMatches, maskTracks, useMC);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return specs;
}
