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

/// @file   emc-reco-workflow.cxx
/// @author Markus Fasel
/// @since  2019-06-07
/// @brief  Basic DPL workflow for EMCAL reconstruction starting from digits (adapted from tpc-reco-workflow.cxx)

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "EMCALWorkflow/RecoWorkflow.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "CommonUtils/ConfigurableParam.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:EMC|emc).*[W,w]riter.*"));
}

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"input-type", o2::framework::VariantType::String, "digits", {"digits, cells, raw, clusters"}},
    {"output-type", o2::framework::VariantType::String, "cells", {"digits, cells, raw, clusters, analysisclusters"}},
    {"enable-digits-printer", o2::framework::VariantType::Bool, false, {"enable digits printer component"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"do not initialize root files readers"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not initialize root file writers"}},
    {"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable sending of MC information"}},
    {"disable-decoding-errors", o2::framework::VariantType::Bool, false, {"disable propagating decoding errors"}},
    {"ignore-dist-stf", o2::framework::VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb EMCAL simulation objects"}},
    {"subspecification", o2::framework::VariantType::Int, 0, {"Subspecification in case the workflow runs in parallel on multiple nodes (i.e. different FLPs)"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

/// The workflow executable for the stand alone EMCAL reconstruction workflow
/// The basic workflow for EMCAL reconstruction is defined in RecoWorkflow.cxx
/// and contains the following default processors
/// - digit reader
/// - clusterer
///
/// The default workflow can be customized by specifying input and output types
/// e.g. digits, raw, tracks.
///
/// MC info is processed by default, disabled by using command line option `--disable-mc`
///
/// This function hooks up the the workflow specifications into the DPL driver.
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& cfgc)
{
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  auto wf = o2::emcal::reco_workflow::getWorkflow(!cfgc.options().get<bool>("disable-mc"),
                                                  !cfgc.options().get<bool>("ignore-dist-stf"),
                                                  cfgc.options().get<bool>("enable-digits-printer"),
                                                  cfgc.options().get<int>("subspecification"),
                                                  cfgc.options().get<std::string>("input-type"),
                                                  cfgc.options().get<std::string>("output-type"),
                                                  cfgc.options().get<bool>("disable-root-input"),
                                                  cfgc.options().get<bool>("disable-root-output"),
                                                  cfgc.options().get<bool>("disable-decoding-errors"),
                                                  cfgc.options().get<bool>("use-ccdb"));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, wf);

  return std::move(wf);
}
