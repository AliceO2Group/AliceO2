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

#include "AlignmentWorkflow/BarrelAlignmentSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/ClusterReaderSpec.h"
#include "TPCWorkflow/ClusterSharingMapSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"

#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

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
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"detectors", VariantType::String, std::string{"ITS,TPC,TRD,TOF"}, {"comma-separated list of detectors"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// the matcher process requires the TPC sector completion to trigger and data on all defined routes
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // the TPC sector completion policy checks when the set of TPC/CLUSTERNATIVE data is complete
  // in addition we require to have input from all other routes
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("barrel-alignment",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  GID::mask_t alowedSources = GID::getSourcesMask("ITS,MFT,TPC,TRD,ITS-TPC,TPC-TOF,TPC-TRD,ITS-TPC-TRD,TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  DetID::mask_t allowedDets = DetID::getMask("ITS,TPC,TRD,TOF,CPV,PHS,EMC,HMP");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2_barrel_alignment_configuration.ini");

  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");

  DetID::mask_t dets = allowedDets & DetID::getMask(configcontext.options().get<std::string>("detectors"));

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  if (GID::includesDet(DetID::TPC, src)) {
    src |= GID::getSourceMask(GID::TPC);
    LOG(info) << "adding TPC request";
  }
  if (GID::includesDet(DetID::TPC, src)) {
    src |= GID::getSourceMask(GID::TPC);
    LOG(info) << "adding TPC request";
  }

  GID::mask_t dummy;
  specs.emplace_back(o2::align::getBarrelAlignmentSpec(src, dets));
  // RS FIXME: check which clusters are really needed
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, src, src, src, false, dummy); // clusters MC is not needed
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, false);

  if (!disableRootOut) {
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
