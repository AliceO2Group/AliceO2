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

#include "GlobalTrackingWorkflow/SecondaryVertexingSpec.h"
#include "GlobalTrackingWorkflow/SecondaryVertexWriterSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/PrimaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
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
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*secondary-vertex-writer.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"vertexing-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use in vertexing"}},
    {"disable-cascade-finder", o2::framework::VariantType::Bool, false, {"do not run cascade finder"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}},
    {"combine-source-devices", o2::framework::VariantType::Bool, false, {"merge DPL source devices"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  GID::mask_t allowedSources = GID::getSourcesMask("ITS,TPC,ITS-TPC,TPC-TOF,TPC-TRD,ITS-TPC-TRD,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2secondary-vertexing-workflow_configuration.ini");
  bool useMC = false;
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto enableCasc = !configcontext.options().get<bool>("disable-cascade-finder");

  GID::mask_t src = allowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("vertexing-sources"));
  GID::mask_t dummy, srcClus = GID::includesDet(DetID::TOF, src) ? GID::getSourceMask(GID::TOF) : dummy; // eventually, TPC clusters will be needed for refit

  WorkflowSpec specs;

  specs.emplace_back(o2::vertexing::getSecondaryVertexingSpec(src, enableCasc));

  // only TOF clusters are needed if TOF is involved, no clusters MC needed
  WorkflowSpec inputspecs;
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, inputspecs, srcClus, src, src, useMC, srcClus);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, inputspecs, useMC); // P-vertex is always needed

  if (configcontext.options().get<bool>("combine-source-devices")) {
    std::vector<DataProcessorSpec> unmerged;
    specs.push_back(specCombiner("SV-Input-Reader", inputspecs, unmerged));
    if (unmerged.size() > 0) {
      // LOG(fatal) << "unexpected DPL merge error";
      for (auto& s : unmerged) {
        specs.push_back(s);
        LOG(info) << " adding unmerged spec " << s.name;
      }
    }
  } else {
    for (auto& s : inputspecs) {
      specs.push_back(s);
    }
  }

  if (!disableRootOut) {
    specs.emplace_back(o2::vertexing::getSecondaryVertexWriterSpec());
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
