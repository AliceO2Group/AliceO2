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
#include "Framework/CompletionPolicyHelpers.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"

#include "GlobalTrackingWorkflow/HMPMatcherSpec.h"
#include "HMPIDWorkflow/HMPMatchedWriterSpec.h"

#include "HMPIDWorkflow/ClustersReaderSpec.h"

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
// #include "TOFBase/Utils.h"
#include "Steer/MCKinematicsReader.h"
#include "TSystem.h"
#include "DetectorsBase/DPLWorkflowUtils.h"

#include "GlobalTracking/MatchHMP.h"

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
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:HMP|hmp).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use: allowed ITS-TPC,TPC-TRD,TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD,ITS-TPC-TRD-TOF (all)"}},
    {"trd-extra-tolerance", o2::framework::VariantType::Float, 0.0f, {"Extra time tolerance for TRD tracks in microsec"}},
    {"tof-extra-tolerance", o2::framework::VariantType::Float, 0.0f, {"Extra time tolerance for TRD tracks in microsec"}},
    {"combine-devices", o2::framework::VariantType::Bool, false, {"merge DPL source/writer devices"}},
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

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto extratolerancetrd = configcontext.options().get<float>("trd-extra-tolerance");
  auto extratolerancetof = configcontext.options().get<float>("tof-extra-tolerance");
  auto disableRootIn = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");

  bool writematching = 0;
  bool writecalib = 0;

  LOG(debug) << "HMP MATCHER WORKFLOW configuration";
  LOG(debug) << "HMP disable-mc = " << configcontext.options().get<std::string>("disable-mc");
  LOG(debug) << "HMP disable-root-input = " << disableRootIn;
  LOG(debug) << "HMP disable-root-output = " << disableRootOut;

  GID::mask_t alowedSources = GID::getSourcesMask("ITS-TPC,TPC-TRD,TPC-TOF,ITS-TPC-TRD,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF");

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));

  GID::mask_t mcmaskcl;
  GID::mask_t nonemask = GID::getSourcesMask(GID::NONE);
  GID::mask_t clustermask = GID::getSourcesMask("HMP");

  if (useMC) {
    mcmaskcl |= GID::getSourceMask(GID::HMP);
  }

  WorkflowSpec inputspecs;
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, inputspecs, clustermask, src, src, useMC, mcmaskcl, GID::getSourcesMask(GID::ALL));
  if (configcontext.options().get<bool>("combine-devices")) {
    std::vector<DataProcessorSpec> unmerged;
    specs.push_back(specCombiner("HMP-readers", inputspecs, unmerged));
    if (unmerged.size() > 0) {
      LOG(fatal) << "Unexpected DPL device merge problem";
    }
  } else {
    for (auto& s : inputspecs) {
      specs.push_back(s);
    }
  }

  specs.emplace_back(o2::globaltracking::getHMPMatcherSpec(src, useMC, extratolerancetrd, extratolerancetof));

  if (!disableRootOut) {
    std::vector<DataProcessorSpec> writers;
    writers.emplace_back(o2::hmpid::getHMPMatchedWriterSpec(useMC, "o2match_hmp.root")); //, false, (int)o2::globaltracking::MatchHMP::trackType::CONSTR, false));

    if (configcontext.options().get<bool>("combine-devices")) {
      std::vector<DataProcessorSpec> unmerged;
      specs.push_back(specCombiner("HMP-writers", writers, unmerged));
      if (unmerged.size() > 0) {
        LOG(fatal) << "Unexpected DPL device merge problem";
      }
    } else {
      for (auto& s : writers) {
        specs.push_back(s);
      }
    }
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
