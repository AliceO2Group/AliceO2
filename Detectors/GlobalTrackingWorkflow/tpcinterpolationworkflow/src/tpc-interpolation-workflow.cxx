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
    {"vtx-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources used for the vertex finding"}},
    {"tracking-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use for track inter-/extrapolation"}},
    {"send-track-data", VariantType::Bool, false, {"Send also the track information to the aggregator"}},
    {"debug-output", VariantType::Bool, false, {"Dump extended tracking information for debugging"}},
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
  GID::mask_t allowedSources = GID::getSourcesMask("ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  GID::mask_t srcVtx = allowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("vtx-sources"));
  GID::mask_t srcTracks = allowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("tracking-sources"));
  if (srcTracks.count() > srcVtx.count()) {
    LOGP(error, "More sources configured for inter-/extrapolation: {} than for vertexing: {}. Additional sources will be ignored", GID::getSourcesNames(srcTracks), GID::getSourcesNames(srcVtx));
    srcTracks &= srcVtx;
  }
  LOG(debug) << "Data sources for inter-/extrapolation: " << GID::getSourcesNames(srcTracks);
  // check first if ITS-TPC tracks were specifically requested from command line
  bool processITSTPConly = srcTracks[GID::ITSTPC];
  srcTracks |= GID::getSourcesMask("ITS,TPC,ITS-TPC"); // now add them in any case
  srcVtx |= srcTracks;
  GID::mask_t srcClusters = srcTracks;
  if (srcTracks[GID::ITSTPCTRD] || srcTracks[GID::ITSTPCTRDTOF]) {
    srcClusters |= GID::getSourcesMask("TRD");
  }
  if (srcTracks[GID::ITSTPCTOF] || srcTracks[GID::ITSTPCTRDTOF]) {
    srcClusters |= GID::getSourcesMask("TOF");
  }
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2tpcinterpolation-workflow_configuration.ini");
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  useMC = false; // force disabling MC as long as it is not implemented
  auto sendTrackData = configcontext.options().get<bool>("send-track-data");
  auto debugOutput = configcontext.options().get<bool>("debug-output");

  specs.emplace_back(o2::tpc::getTPCInterpolationSpec(srcClusters, srcVtx, srcTracks, useMC, processITSTPConly, sendTrackData, debugOutput));
  if (!configcontext.options().get<bool>("disable-root-output")) {
    specs.emplace_back(o2::tpc::getTPCResidualWriterSpec(sendTrackData, debugOutput));
  }

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcClusters, srcVtx, srcVtx, useMC);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC); // P-vertex is always needed

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return specs;
}
