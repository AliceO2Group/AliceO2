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
/// \brief This file defines the workflow for generating the pid signal.
/// \author Felix Schlepper

#include "TRDWorkflow/TRDPIDSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"

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
    {"disable-mc", o2::framework::VariantType::Bool, false, {"Disable MC labels"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GTrackID::ALL}, {"comma-separated list of sources to use for tracking"}},
  };
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  GTrackID::mask_t allowedSources = GTrackID::getSourcesMask("TRD");
  GTrackID::mask_t srcTRD = allowedSources & GTrackID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));

  // MC labels are passed through for the global tracking downstream
  // in case ROOT output is requested the tracklet labels are duplicated
  bool useMC = !configcontext.options().get<bool>("disable-mc");

  if (!configcontext.options().get<bool>("disable-root-input")) {
    // TODO
    // spec.emplace_back(o2::trd::getTRDTrackletReaderSpec(useMC, false));
  }

  // TODO
  specs.emplace_back(o2::trd::getTRDPIDSpec(srcTRD));

  if (!configcontext.options().get<bool>("disable-root-output")) {
    // TODO
    // spec.emplace_back(o2::trd::getTRDCalibratedTrackletWriterSpec(useMC));
  }

  // input
  auto maskMatches = GTrackID::getSourcesMask(GTrackID::NONE);
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcTRD, maskMatches, srcTRD, useMC);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return specs;
}
