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
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"

#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/NoInpDummyOutSpec.h"
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
    {"enable-mc", o2::framework::VariantType::Bool, false, {"enable MC-info checks"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"detectors", VariantType::String, std::string{"ITS,TPC,TRD,TOF"}, {"comma-separated list of detectors"}},
    {"enable-tpc-tracks", VariantType::Bool, false, {"allow reading TPC tracks"}},
    {"enable-tpc-clusters", VariantType::Bool, false, {"allow reading TPC clusters (will trigger TPC tracks reading)"}},
    {"enable-cosmic", VariantType::Bool, false, {"enable cosmic tracks)"}},
    {"postprocessing", VariantType::Int, 0, {"postprocessing bits: 1 - extract alignment objects, 2 - check constraints, 4 - print mpParams/Constraints, 8 - relabel pede results"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::tpc::CorrectionMapsLoader::addGlobalOptions(options);
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
  GID::mask_t alowedSources = GID::getSourcesMask("ITS,TPC,TRD,ITS-TPC,TPC-TOF,TPC-TRD,ITS-TPC-TRD,TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  DetID::mask_t allowedDets = DetID::getMask("ITS,TPC,TRD,TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  int postprocess = configcontext.options().get<int>("postprocessing");

  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  bool loadTPCClusters = configcontext.options().get<bool>("enable-tpc-clusters");
  bool loadTPCTracks = configcontext.options().get<bool>("enable-tpc-tracks");
  bool enableCosmic = configcontext.options().get<bool>("enable-cosmic");
  bool useMC = configcontext.options().get<bool>("enable-mc");

  DetID::mask_t dets = allowedDets & DetID::getMask(configcontext.options().get<std::string>("detectors"));
  DetID::mask_t skipDetClusters; // optionally skip automatically loaded clusters
  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  GID::mask_t srcCl{}, srcMP = src; // we may need to load more track types than requested to satisfy dependencies, but only those will be fed to millipede

  if (dets[DetID::TPC]) {
    loadTPCClusters = loadTPCTracks = true;
  }
  auto sclOpt = o2::tpc::CorrectionMapsLoader::parseGlobalOptions(configcontext.options());
  if (!postprocess) { // this part is needed only if the data should be read
    if (GID::includesDet(DetID::ITS, src)) {
      src |= GID::getSourceMask(GID::ITS);
      srcCl |= GID::getSourceMask(GID::ITS);
      LOG(info) << "adding ITS request";
    }

    if (GID::includesDet(DetID::TPC, src)) {
      if (loadTPCTracks || loadTPCClusters) {
        src |= GID::getSourceMask(GID::TPC);
        LOG(info) << "adding TPC request";
      }
      if (loadTPCClusters) {
        srcCl |= GID::getSourceMask(GID::TPC);
      } else {
        skipDetClusters |= DetID::getMask(DetID::TPC);
        LOG(info) << "Skipping TPC clusters";
      }
    }
    if (GID::includesDet(DetID::TRD, src)) {
      src |= GID::getSourceMask(GID::TRD);
      srcCl |= GID::getSourceMask(GID::TRD);
      if (GID::includesDet(DetID::ITS, src)) {
        src |= GID::getSourceMask(GID::ITSTPC);
      }
      LOG(info) << "adding TRD request";
    }
    if (GID::includesDet(DetID::TOF, src)) {
      src |= GID::getSourceMask(GID::TOF);
      srcCl |= GID::getSourceMask(GID::TOF);
      if (GID::includesDet(DetID::ITS, src)) {
        src |= GID::getSourceMask(GID::ITSTPC);
      }
      if (GID::includesDet(DetID::TRD, src)) {
        src |= GID::getSourceMask(GID::ITSTPCTRD);
      }
      LOG(info) << "adding TOF request";
    }
    if (sclOpt.lumiType == 1) {
      src = src | GID::getSourcesMask("CTP");
    }
    // write the configuration used for the workflow
    o2::conf::ConfigurableParam::writeINI("o2_barrel_alignment_configuration.ini");
  }

  specs.emplace_back(o2::align::getBarrelAlignmentSpec(srcMP, src, dets, skipDetClusters, enableCosmic, postprocess, useMC, sclOpt));
  // RS FIXME: check which clusters are really needed
  if (!postprocess) {
    GID::mask_t dummy;
    o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcCl, src, src, useMC, dummy); // clusters MC is not needed
    o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC);
    if (enableCosmic) {
      o2::globaltracking::InputHelper::addInputSpecsCosmics(configcontext, specs, useMC);
    }
  } else { // add dummy driver
    specs.emplace_back(o2::globaltracking::getNoInpDummyOutSpec(0));
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);
  return std::move(specs);
}
