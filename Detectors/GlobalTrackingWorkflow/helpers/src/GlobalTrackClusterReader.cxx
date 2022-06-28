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
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigContext.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using namespace o2::dataformats;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable visualization of MC data"}},
    {"track-types", VariantType::String, std::string{GlobalTrackID::NONE}, {"comma-separated list of track sources to read"}},
    {"cluster-types", VariantType::String, std::string{GlobalTrackID::NONE}, {"comma-separated list of cluster sources to read"}},
    {"primary-vertices", VariantType::Bool, false, {"read primary vertices"}},
    {"secondary-vertices", VariantType::Bool, false, {"read secondary vertices"}},
    {"cosmic", VariantType::Bool, false, {"read cosmic tracks"}},
    {"ir-frames-its", VariantType::Bool, false, {"read ITS IR frames"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable reading root files, essentially making this workflow void, but needed for compatibility"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  bool useMC = !cfgc.options().get<bool>("disable-mc");
  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("track-types"));
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("cluster-types"));
  bool pv = cfgc.options().get<bool>("primary-vertices");
  bool sv = cfgc.options().get<bool>("secondary-vertices");
  bool cosm = cfgc.options().get<bool>("cosmic");
  bool irits = cfgc.options().get<bool>("ir-frames-its");

  if (!cfgc.helpOnCommandLine() && srcTrk.none() && srcCl.none() && !(pv || sv || cosm || irits)) {
    throw std::runtime_error("nothing requested to read");
  }
  auto srcMtc = srcTrk & ~GlobalTrackID::getSourceMask(GlobalTrackID::MFTMCH); // Do not request MFTMCH matches
  InputHelper::addInputSpecs(cfgc, specs, srcCl, srcMtc, srcTrk, useMC);
  if (pv) {
    InputHelper::addInputSpecsPVertex(cfgc, specs, useMC);
  }
  if (sv) {
    InputHelper::addInputSpecsSVertex(cfgc, specs);
  }
  if (cosm) {
    InputHelper::addInputSpecsCosmics(cfgc, specs, useMC);
  }
  if (irits) {
    InputHelper::addInputSpecsIRFramesITS(cfgc, specs);
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return std::move(specs);
}
