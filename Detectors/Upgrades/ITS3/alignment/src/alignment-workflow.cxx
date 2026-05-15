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
#include "Framework/ConfigParamSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "GlobalTrackingWorkflowHelpers/NoInpDummyOutSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DataFormatsITSMFT/DPLAlpideParamInitializer.h"
#include "ITS3Align/AlignmentSpec.h"

using namespace o2::framework;
using namespace o2::its3::align;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"enable MC propagation"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of track sources to use"}},
    {"cluster-sources", VariantType::String, "ITS", {"comma-separated list of cluster sources to use"}},
    {"with-its", VariantType::Bool, false, {"ITS alignment mode"}},
    {"without-pv", VariantType::Bool, false, {"Do not use in track refit the PV as an additional constraint"}},
    {"output", VariantType::String, "", {"output steering"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  o2::itsmft::DPLAlpideParamInitializer::addITSConfigOption(options);
  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  o2::conf::ConfigurableParam::updateFromString(cfg.options().get<std::string>("configKeyValues"));
  const GID::mask_t allowedSourcesTrc = GID::getSourcesMask("ITS,TPC,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  const GID::mask_t allowedSourcesClus = GID::getSourcesMask("ITS");
  const GID::mask_t srcTrc = allowedSourcesTrc & GID::getSourcesMask(cfg.options().get<std::string>("track-sources"));
  const GID::mask_t srcCls = allowedSourcesClus & GID::getSourcesMask(cfg.options().get<std::string>("cluster-sources"));
  const auto useMC = !cfg.options().get<bool>("disable-mc");
  const auto withPV = !cfg.options().get<bool>("without-pv");
  const auto withITS = cfg.options().get<bool>("with-its");
  const OutputEnum output(cfg.options().get<std::string>("output"));

  WorkflowSpec specs;
  if (!output[OutputOpt::MilleRes]) {
    o2::globaltracking::InputHelper::addInputSpecs(cfg, specs, srcCls, srcTrc, srcTrc, useMC);
    if (withPV && !useMC) {
      o2::globaltracking::InputHelper::addInputSpecsPVertex(cfg, specs, useMC);
    }
  } else {
    specs.emplace_back(o2::globaltracking::getNoInpDummyOutSpec(0));
  }

  specs.emplace_back(o2::its3::align::getAlignmentSpec(srcTrc, srcCls, useMC, withPV, withITS, output));

  o2::raw::HBFUtilsInitializer hbfIni(cfg, specs);
  return std::move(specs);
}
