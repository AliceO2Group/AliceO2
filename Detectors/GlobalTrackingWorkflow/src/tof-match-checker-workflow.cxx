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
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "GlobalTrackingWorkflow/TOFMatchChecker.h"
#include "TOFWorkflowIO/TOFMatchedWriterSpec.h"
#include "TOFWorkflowIO/TOFCalibWriterSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using GID = o2::dataformats::GlobalTrackID;
// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:ITS|its).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use: allowed TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF (all)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2match-check-tof-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");

  LOG(debug) << "TOF MATCHER WORKFLOW configuration";
  LOG(debug) << "TOF track inputs = " << configcontext.options().get<std::string>("track-sources");
  LOG(debug) << "TOF disable-mc = " << configcontext.options().get<std::string>("disable-mc");

  GID::mask_t alowedSources = GID::getSourcesMask("ITS-TPC-TOF,TPC-TOF,ITS-TPC-TRD-TOF,TPC-TRD-TOF");

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  GID::mask_t mcmaskcl;
  GID::mask_t nonemask = GID::getSourcesMask(GID::NONE);
  GID::mask_t clustermask = GID::getSourcesMask("TOF");

  if (useMC) {
    mcmaskcl |= GID::getSourceMask(GID::TOF);
  }
  // pass strict flag to fetch eventual TPC-TRD input with correct subspec
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, clustermask, src, src, useMC, mcmaskcl, GID::getSourcesMask(GID::ALL), false);

  specs.emplace_back(o2::globaltracking::getTOFMatchCheckerSpec(src, useMC)); // doTPCrefit not yet supported (need to load TPC clusters?)

  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
