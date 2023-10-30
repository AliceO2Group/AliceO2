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
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "TRDWorkflowIO/TRDCalibWriterSpec.h"
#include "TRDPID/PIDBase.h"
#include "TRDWorkflowIO/TRDTrackWriterSpec.h"
#include "TRDWorkflowIO/TRDDigitReaderSpec.h"
#include "TRDWorkflow/TrackBasedCalibSpec.h"
#include "TRDWorkflow/TRDGlobalTrackingSpec.h"
#include "TRDWorkflow/TRDGlobalTrackingQCSpec.h"
#include "TRDWorkflow/TRDPulseHeightSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

using namespace o2::framework;
using GTrackID = o2::dataformats::GlobalTrackID;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRD|trd).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", VariantType::Bool, false, {"Disable MC labels"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}},
    {"enable-vdexb-calib", VariantType::Bool, false, {"enable vDrift and ExB calibration based on tracking output"}},
    {"enable-gain-calib", VariantType::Bool, false, {"enable collection of dEdx histos for gain calibration"}},
    {"enable-qc", VariantType::Bool, false, {"enable tracking QC"}},
    {"enable-pid", VariantType::Bool, false, {"Enable PID"}},
    {"enable-ph", VariantType::Bool, false, {"Enable creation of PH plots"}},
    {"trd-digits-spec", VariantType::Int, 0, {"Input digits subspec, ignored if disable-root-input is false"}},
    {"track-sources", VariantType::String, std::string{GTrackID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"filter-trigrec", VariantType::Bool, false, {"ignore interaction records without ITS data"}},
    {"strict-matching", VariantType::Bool, false, {"High purity preliminary matching"}},
    {"disable-ft0-pileup-tagging", VariantType::Bool, false, {"Do not request FT0 for pile-up determination"}},
    {"require-ctp-lumi", o2::framework::VariantType::Bool, false, {"require CTP lumi for TPC correction scaling"}},
    {"policy", VariantType::String, "default", {"Pick PID policy (=default)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  GTrackID::mask_t allowedSources = GTrackID::getSourcesMask("ITS-TPC,TPC");
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2trdtracking-workflow_configuration.ini");
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto pid = configcontext.options().get<bool>("enable-pid");
  auto strict = configcontext.options().get<bool>("strict-matching");
  auto trigRecFilterActive = configcontext.options().get<bool>("filter-trigrec");
  auto requireCTPLumi = configcontext.options().get<bool>("require-ctp-lumi");
  auto vdexb = configcontext.options().get<bool>("enable-vdexb-calib");
  auto gain = configcontext.options().get<bool>("enable-gain-calib");
  auto pulseHeight = configcontext.options().get<bool>("enable-ph");
  auto digitsSpec = configcontext.options().get<int>("trd-digits-spec");
  bool rootInput = !configcontext.options().get<bool>("disable-root-input");
  GTrackID::mask_t srcTRD = allowedSources & GTrackID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  if (strict && (srcTRD & ~GTrackID::getSourcesMask("TPC")).any()) {
    LOGP(warning, "In strict matching mode only TPC source allowed, {} asked, redefining", GTrackID::getSourcesNames(srcTRD));
    srcTRD = GTrackID::getSourcesMask("TPC");
  }
  if (!configcontext.options().get<bool>("disable-ft0-pileup-tagging")) {
    srcTRD |= GTrackID::getSourcesMask("FT0");
  }
  if (requireCTPLumi) {
    srcTRD = srcTRD | GTrackID::getSourcesMask("CTP");
  }
  // Parse PID policy string
  o2::trd::PIDPolicy policy{o2::trd::PIDPolicy::DEFAULT};
  if (pid) {
    auto policyStr = configcontext.options().get<std::string>("policy");
    auto policyIt = o2::trd::PIDPolicyString.find(policyStr);
    if (policyIt == o2::trd::PIDPolicyString.end()) {
      throw std::runtime_error(fmt::format("No PID model named {:s} available!", policyStr));
    }
    policy = policyIt->second;
    LOGF(info, "Using PID policy %s(%u)", policyStr, static_cast<unsigned int>(policy));
  }

  // processing devices
  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::trd::getTRDGlobalTrackingSpec(useMC, srcTRD, trigRecFilterActive, strict, pid, policy));
  if (vdexb || gain) {
    specs.emplace_back(o2::trd::getTRDTrackBasedCalibSpec(srcTRD, vdexb, gain));
  }
  if (configcontext.options().get<bool>("enable-qc")) {
    specs.emplace_back(o2::trd::getTRDGlobalTrackingQCSpec(srcTRD));
  }
  if (pulseHeight) {
    if (rootInput) {
      specs.emplace_back(o2::trd::getTRDDigitReaderSpec(useMC));
    }
    specs.emplace_back(o2::framework::getTRDPulseHeightSpec(srcTRD, rootInput ? 1 : digitsSpec));
  }

  // output devices
  if (!configcontext.options().get<bool>("disable-root-output")) {
    if (GTrackID::includesSource(GTrackID::Source::ITSTPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDGlobalTrackWriterSpec(useMC));
    }
    if (GTrackID::includesSource(GTrackID::Source::TPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDTPCTrackWriterSpec(useMC, strict));
    }
    if (vdexb || gain || pulseHeight) {
      specs.emplace_back(o2::trd::getTRDCalibWriterSpec(vdexb, gain, pulseHeight));
    }
    if (configcontext.options().get<bool>("enable-qc")) {
      specs.emplace_back(o2::trd::getTRDTrackingQCWriterSpec());
    }
  }

  // input
  auto maskClusters = GTrackID::getSourcesMask("TRD,TPC");
  auto maskTracks = srcTRD | GTrackID::getSourcesMask("TPC"); // we always need the TPC tracks for the refit
  if (GTrackID::includesDet(GTrackID::DetID::ITS, srcTRD)) {
    maskClusters |= GTrackID::getSourcesMask("ITS");
    maskTracks |= GTrackID::getSourcesMask("ITS");
  }
  auto maskMatches = GTrackID::getSourcesMask(GTrackID::NONE);
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, maskClusters, maskMatches, maskTracks, useMC);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return specs;
}
