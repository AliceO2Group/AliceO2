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
#include "GlobalTrackingWorkflow/TOFEventTimeChecker.h"
#include "TOFWorkflowIO/TOFMatchedWriterSpec.h"
#include "TOFWorkflowIO/TOFCalibWriterSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "TOFBase/Utils.h"
#include "Steer/MCKinematicsReader.h"

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
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use: allowed TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF (all)"}},
    {"event-time-spread", o2::framework::VariantType::Float, 200.0f, {"Event time spread"}},
    {"lhc-phase", o2::framework::VariantType::Float, 0.0f, {"LHCp phase"}},
    {"eta-min", o2::framework::VariantType::Float, -0.8f, {"min tof eta"}},
    {"eta-max", o2::framework::VariantType::Float, 0.8f, {"max tof eta"}},
    {"fill-mask", o2::framework::VariantType::String, "", {"fill scheme, collision bunches"}},
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
  o2::conf::ConfigurableParam::writeINI("o2eventtime-check-tof-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");

  auto spread = configcontext.options().get<float>("event-time-spread");
  auto phase = configcontext.options().get<float>("lhc-phase");
  auto minEta = configcontext.options().get<float>("eta-min");
  auto maxEta = configcontext.options().get<float>("eta-max");
  auto fillscheme = configcontext.options().get<std::string>("fill-mask");

  o2::tof::Utils::mEventTimeSpread = spread;
  o2::tof::Utils::mLHCPhase = phase;
  o2::tof::Utils::mEtaMin = minEta;
  o2::tof::Utils::mEtaMax = maxEta;

  while (fillscheme.size()) {
    int bc;
    int res = sscanf(fillscheme.c_str(), "%d", &bc);
    if (res) {
      o2::tof::Utils::addInteractionBC(bc);
    } else {
      fillscheme.clear();
    }
    int next = fillscheme.find(",");
    if (next < fillscheme.size()) {
      fillscheme.erase(0, next + 1);
    } else {
      fillscheme.clear();
    }
  }

  LOG(debug) << "TOF EVENTTIME CHECKER WORKFLOW configuration";
  LOG(debug) << "TOF track inputs = " << configcontext.options().get<std::string>("track-sources");

  GID::mask_t alowedSources = GID::getSourcesMask("ITS-TPC-TOF,TPC-TOF,ITS-TPC-TRD-TOF,TPC-TRD-TOF,ITS-TPC,TPC,ITS-TPC-TRD,TPC-TRD");

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  GID::mask_t mcmaskcl;
  GID::mask_t nonemask = GID::getSourcesMask(GID::NONE);
  GID::mask_t clustermask = GID::getSourcesMask("TOF");

  if (useMC) {
    mcmaskcl |= GID::getSourceMask(GID::TOF);
  }
  // pass strict flag to fetch eventual TPC-TRD input with correct subspec
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, clustermask, src, src, useMC, mcmaskcl, GID::getSourcesMask(GID::ALL), false);

  specs.emplace_back(o2::globaltracking::getTOFEventTimeCheckerSpec(src, useMC)); // doTPCrefit not yet supported (need to load TPC clusters?)

  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
