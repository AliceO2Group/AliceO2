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
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "TPCInterpolationWorkflow/TPCInterpolationSpec.h"
#include "TPCInterpolationWorkflow/TPCResidualWriterSpec.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}},
    {"disable-mc", VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"enable-itsonly", VariantType::Bool, false, {"process tracks without outer point (ITS-TPC only)"}},
    {"tracking-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"enable-residual-writer", VariantType::Bool, false, {"write unbinned and unfiltered residuals"}},
    {"send-track-data", VariantType::Bool, false, {"Send also the track information to the aggregator"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// the matcher process requires the TPC sector completion to trigger and data on
// all defined routes
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // the TPC sector completion policy checks when the set of TPC/CLUSTERNATIVE data is complete
  // in addition we require to have input from all other routes
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("tpc-track-interpolation",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*tpc-residuals-writer.*"));
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  GID::mask_t allowedSources = GID::getSourcesMask("ITS,TPC,TRD,TOF,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2tpcinterpolation-workflow_configuration.ini");
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  useMC = false; // force disabling MC as long as it is not implemented
  auto processITSTPConly = configcontext.options().get<bool>("enable-itsonly");
  auto writeResiduals = configcontext.options().get<bool>("enable-residual-writer");
  auto sendTrackData = configcontext.options().get<bool>("send-track-data");
  GID::mask_t src = allowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("tracking-sources"));
  LOG(info) << "Data sources: " << GID::getSourcesNames(src);

  specs.emplace_back(o2::tpc::getTPCInterpolationSpec(src, useMC, processITSTPConly, writeResiduals, sendTrackData));
  if (!configcontext.options().get<bool>("disable-root-output") && writeResiduals) {
    specs.emplace_back(o2::tpc::getTPCResidualWriterSpec(useMC));
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, src, src, src, useMC);

  return specs;
}
