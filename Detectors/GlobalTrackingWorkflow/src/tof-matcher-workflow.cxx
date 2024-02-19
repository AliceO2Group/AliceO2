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
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "GlobalTrackingWorkflow/TOFMatcherSpec.h"
#include "TOFWorkflowIO/TOFMatchedWriterSpec.h"
#include "TOFWorkflowIO/TOFCalibWriterSpec.h"
#include "TOFWorkflowIO/TOFMatchableWriterSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "TOFBase/Utils.h"
#include "Steer/MCKinematicsReader.h"
#include "TSystem.h"
#include "DetectorsBase/DPLWorkflowUtils.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "TPCWorkflow/TPCScalerSpec.h"

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
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TOF|tof).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"tof-lanes", o2::framework::VariantType::Int, 1, {"number of parallel lanes up to the matcher"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use: allowed TPC,ITS-TPC,TPC-TRD,ITS-TPC-TRD (all)"}},
    {"use-fit", o2::framework::VariantType::Bool, false, {"enable access to fit info for calibration"}},
    {"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}},
    {"strict-matching", o2::framework::VariantType::Bool, false, {"High purity preliminary matching"}},
    {"output-type", o2::framework::VariantType::String, "matching-info", {"matching-info, calib-info"}},
    {"enable-dia", o2::framework::VariantType::Bool, false, {"to require diagnostic freq and then write to calib outputs (obsolete since now default)"}},
    {"trd-extra-tolerance", o2::framework::VariantType::Float, 500.0f, {"Extra time tolerance for TRD tracks in ns"}},
    {"refit-tpc-tof", o2::framework::VariantType::Bool, false, {"Refit unconstrained TPC tracks matched to TOF (if false - just move)"}},
    {"write-matchable", o2::framework::VariantType::Bool, false, {"write all matchable pairs in a file (o2matchable_tof.root)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}},
    {"combine-devices", o2::framework::VariantType::Bool, false, {"merge DPL source/writer devices"}}};
  o2::tpc::CorrectionMapsLoader::addGlobalOptions(options);
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
  o2::conf::ConfigurableParam::writeINI("o2match-tof-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootIn = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto useFIT = configcontext.options().get<bool>("use-fit");
  auto useCCDB = configcontext.options().get<bool>("use-ccdb");
  auto strict = configcontext.options().get<bool>("strict-matching");
  auto diagnostic = configcontext.options().get<bool>("enable-dia");
  auto extratolerancetrd = configcontext.options().get<float>("trd-extra-tolerance");
  auto writeMatchable = configcontext.options().get<bool>("write-matchable");
  auto sclOpt = o2::tpc::CorrectionMapsLoader::parseGlobalOptions(configcontext.options());
  bool writematching = 0;
  bool writecalib = 0;
  bool refitTPCTOF = configcontext.options().get<bool>("refit-tpc-tof");
  auto outputType = configcontext.options().get<std::string>("output-type");
  auto nLanes = configcontext.options().get<int>("tof-lanes");
  if (outputType.rfind("matching-info") < outputType.size()) {
    writematching = 1;
  }
  if (outputType.rfind("calib-info") < outputType.size()) {
    writecalib = 1;
    if (!diagnostic) {
      diagnostic = true;
      LOG(info) << "Diagnostic switched on since required for calibInfo time";
    }
  }

  if (!writecalib) {
    useFIT = false;
  }

  LOG(debug) << "TOF MATCHER WORKFLOW configuration";
  LOG(debug) << "TOF track inputs = " << configcontext.options().get<std::string>("track-sources");
  LOG(debug) << "TOF output = " << outputType;
  LOG(debug) << "TOF disable-mc = " << configcontext.options().get<std::string>("disable-mc");
  LOG(debug) << "TOF use-ccdb = " << useCCDB;
  LOG(debug) << "TOF use-fit = " << useFIT;
  LOG(debug) << "TOF disable-root-input = " << disableRootIn;
  LOG(debug) << "TOF disable-root-output = " << disableRootOut;
  LOG(debug) << "TOF matching in strict mode = " << strict;
  LOG(debug) << "TOF extra time tolerance for TRD tracks = " << extratolerancetrd;
  LOG(debug) << "Store all matchables = " << writeMatchable;
  LOG(debug) << "TOF Nlanes for matcher = " << nLanes;

  //GID::mask_t alowedSources = GID::getSourcesMask("TPC,ITS-TPC");
  GID::mask_t alowedSources = GID::getSourcesMask("TPC,ITS-TPC,TPC-TRD,ITS-TPC-TRD");

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  if (strict && (src & ~GID::getSourcesMask("TPC,TPC-TRD")).any()) {
    LOGP(warning, "In strict matching mode only TPC and TPC-TRD sources allowed, {} asked, redefining", GID::getSourcesNames(src));
    src &= GID::getSourcesMask("TPC,TPC-TRD");
  }
  GID::mask_t mcmaskcl;
  GID::mask_t nonemask = GID::getSourcesMask(GID::NONE);
  GID::mask_t clustermask = GID::getSourcesMask("TOF");
  if (useFIT) {
    clustermask |= GID::getSourceMask(GID::FT0);
  }
  if (src[GID::TPC] && refitTPCTOF) { // load clusters
    clustermask |= GID::getSourceMask(GID::TPC);
  }
  if (sclOpt.requestCTPLumi) {
    src = src | GID::getSourcesMask("CTP");
  }
  if (useMC) {
    mcmaskcl |= GID::getSourceMask(GID::TOF);
  }
  // pass strict flag to fetch eventual TPC-TRD input with correct subspec
  WorkflowSpec inputspecs;
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, inputspecs, clustermask, nonemask, src, useMC, mcmaskcl, GID::getSourcesMask(GID::ALL), strict);
  if (configcontext.options().get<bool>("combine-devices")) {
    std::vector<DataProcessorSpec> unmerged;
    specs.push_back(specCombiner("TOF-readers", inputspecs, unmerged));
    if (unmerged.size() > 0) {
      LOG(fatal) << "Unexpected DPL device merge problem";
    }
  } else {
    for (auto& s : inputspecs) {
      specs.push_back(s);
    }
  }
  if (sclOpt.needTPCScalersWorkflow() && !configcontext.options().get<bool>("disable-root-input")) {
    specs.emplace_back(o2::tpc::getTPCScalerSpec(sclOpt.lumiType == 2, sclOpt.enableMShapeCorrection));
  }
  specs.emplace_back(o2::globaltracking::getTOFMatcherSpec(src, useMC, useFIT, refitTPCTOF, strict, extratolerancetrd, writeMatchable, sclOpt, nLanes)); // doTPCrefit not yet supported (need to load TPC clusters?)

  if (!disableRootOut) {
    std::vector<DataProcessorSpec> writers;
    if (writematching) {
      if (GID::includesSource(GID::TPC, src)) { // matching to TPC was requested
        writers.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC, "o2match_tof_tpc.root", true, (int)o2::dataformats::MatchInfoTOFReco::TrackType::TPC, strict));
      }
      if (GID::includesSource(GID::ITSTPC, src)) { // matching to ITS-TPC was requested, there is not strict mode in this case
        writers.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC, "o2match_tof_itstpc.root", false, (int)o2::dataformats::MatchInfoTOFReco::TrackType::ITSTPC, false));
      }
      if (GID::includesSource(GID::TPCTRD, src)) { // matching to TPC-TRD was requested, there is not strict mode in this case
        writers.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC, "o2match_tof_tpctrd.root", false, (int)o2::dataformats::MatchInfoTOFReco::TrackType::TPCTRD, strict));
      }
      if (GID::includesSource(GID::ITSTPCTRD, src)) { // matching to ITS-TPC-TRD was requested, there is not strict mode in this case
        writers.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC, "o2match_tof_itstpctrd.root", false, (int)o2::dataformats::MatchInfoTOFReco::TrackType::ITSTPCTRD, false));
      }
    }
    if (writecalib) {
      writers.emplace_back(o2::tof::getTOFCalibWriterSpec("o2calib_tof.root", 0, diagnostic));
    }

    if (writeMatchable) {
      specs.emplace_back(o2::tof::getTOFMatchableWriterSpec("o2matchable_tof.root"));
    }

    if (configcontext.options().get<bool>("combine-devices")) {
      std::vector<DataProcessorSpec> unmerged;
      specs.push_back(specCombiner("TOF-writers", writers, unmerged));
      if (unmerged.size() > 0) {
        LOG(fatal) << "Unexpected DPL device merge problem";
      }
    } else {
      for (auto& s : writers) {
        specs.push_back(s);
      }
    }
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
