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

#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "GlobalTrackingWorkflow/PrimaryVertexWriterSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/GlobalFwdTrackReaderSpec.h"
#include "GlobalTrackingWorkflow/VertexTrackMatcherSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "FT0Workflow/RecPointReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "DetectorsBase/DPLWorkflowUtils.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;
// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*primary-vertex-writer.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"vertexing-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use in vertexing"}},
    {"skip", VariantType::Bool, false, {"pass-through mode (skip vertexing)"}},
    {"validate-with-ft0", o2::framework::VariantType::Bool, false, {"use FT0 time for vertex validation"}},
    {"vertex-track-matching-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use in vertex-track associations or \"none\" to disable matching"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}},
    {"combine-source-devices", o2::framework::VariantType::Bool, false, {"merge DPL source devices"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  GID::mask_t allowedSourcesPV = GID::getSourcesMask("ITS,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  GID::mask_t allowedSourcesVT = GID::getSourcesMask("ITS,MFT,TPC,MCH,MCH-MID,MFT-MCH,ITS-TPC,TPC-TOF,TPC-TRD,ITS-TPC-TRD,TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC-TRD-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2primary-vertexing-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto validateWithFT0 = configcontext.options().get<bool>("validate-with-ft0");
  auto skip = configcontext.options().get<bool>("skip");

  GID::mask_t srcPV = allowedSourcesPV & GID::getSourcesMask(configcontext.options().get<std::string>("vertexing-sources"));
  GID::mask_t srcVT = allowedSourcesVT & GID::getSourcesMask(configcontext.options().get<std::string>("vertex-track-matching-sources"));
  if (validateWithFT0) {
    srcPV |= GID::getSourceMask(GID::FT0);
  }
  GID::mask_t srcComb = srcPV | srcVT;
  GID::mask_t dummy, srcClus = GID::includesDet(DetID::TOF, srcComb) ? GID::getSourceMask(GID::TOF) : dummy;

  specs.emplace_back(o2::vertexing::getPrimaryVertexingSpec(srcPV, skip, validateWithFT0, useMC));
  specs.emplace_back(o2::vertexing::getVertexTrackMatcherSpec(srcVT));

  auto srcMtc = srcComb & ~GID::getSourceMask(GID::MFTMCH); // Do not request MFTMCH matches

  // only TOF clusters are needed if TOF is involved, no clusters MC needed
  WorkflowSpec inputspecs;
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, inputspecs, srcClus, srcMtc, srcComb, useMC, dummy);

  if (configcontext.options().get<bool>("combine-source-devices")) {
    std::vector<DataProcessorSpec> unmerged;
    specs.push_back(specCombiner("PV-Input-Reader", inputspecs, unmerged));
    // Note: Switch in future to core framework function, once it can handle duplicate options
    // auto combined = o2::framework::workflow::combine("PV-Input-Reader", inputspecs, true);
    // for (auto& c : combined) {
    //  specs.push_back(c);
    // }
    if (unmerged.size() > 0) {
      LOG(fatal) << "unexpected DPL merge error";
    }
  } else {
    for (auto& s : inputspecs) {
      specs.push_back(s);
    }
  }

  if (!disableRootOut) {
    specs.emplace_back(o2::vertexing::getPrimaryVertexWriterSpec(srcVT.none(), useMC));
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  if (srcPV.any() && !configcontext.options().get<bool>("disable-root-output")) {
    o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);
  }
  return std::move(specs);
}
