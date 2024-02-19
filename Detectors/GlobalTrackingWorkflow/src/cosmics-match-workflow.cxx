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
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/ClusterReaderSpec.h"
#include "TPCWorkflow/TPCScalerSpec.h"
#include "TPCWorkflow/ClusterSharingMapSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/CosmicsMatchingSpec.h"
#include "GlobalTrackingWorkflow/TrackCosmicsWriterSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;
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
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::tpc::CorrectionMapsLoader::addGlobalOptions(options);
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// RS FIXME Is this needed ?
// the matcher process requires the TPC sector completion to trigger and data on
// all defined routes
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // the TPC sector completion policy checks when the set of TPC/CLUSTERNATIVE data is complete
  // in addition we require to have input from all other routes
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("cosmics-matcher",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*cosm.*[W,w]riter.*"));
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  GID::mask_t alowedSources = GID::getSourcesMask("ITS,TPC,ITS-TPC,TPC-TRD,TPC-TOF,TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC-TRD-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2match-cosmics-workflow_configuration.ini");
  auto sclOpt = o2::tpc::CorrectionMapsLoader::parseGlobalOptions(configcontext.options());
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  if (GID::includesDet(DetID::TPC, src)) {
    src |= GID::getSourceMask(GID::TPC);
  }
  if (GID::includesDet(DetID::TRD, src)) {
    src |= GID::getSourceMask(GID::TRD);
  }
  if (GID::includesDet(DetID::TOF, src)) {
    src |= GID::getSourceMask(GID::TOF);
  }
  if (sclOpt.requestCTPLumi) {
    src = src | GID::getSourcesMask("CTP");
  }
  GID::mask_t srcCl = src;
  GID::mask_t dummy;
  if (sclOpt.needTPCScalersWorkflow() && !configcontext.options().get<bool>("disable-root-input")) {
    specs.emplace_back(o2::tpc::getTPCScalerSpec(sclOpt.lumiType == 2, sclOpt.enableMShapeCorrection));
  }
  specs.emplace_back(o2::globaltracking::getCosmicsMatchingSpec(src, useMC, sclOpt));

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, src, src, src, useMC, dummy); // clusters MC is not needed

  if (!disableRootOut) {
    specs.emplace_back(o2::globaltracking::getTrackCosmicsWriterSpec(useMC));
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
