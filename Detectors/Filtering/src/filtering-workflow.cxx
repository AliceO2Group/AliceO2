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

#include "FilteringSpec.h"
#include "Framework/CompletionPolicy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"disable-secondary-vertices", o2::framework::VariantType::Bool, false, {"disable filling secondary vertices"}},
    {"data-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  bool enableSV = !configcontext.options().get<bool>("disable-secondary-vertices");
  GID::mask_t allowedSrc = GID::getSourcesMask("ITS,TPC,ITS-TPC,ITS-TPC-TOF,TPC-TOF,FT0,FV0,FDD,TPC-TRD,ITS-TPC-TRD,FT0,FV0,FDD,CTP");
  GID::mask_t src = allowedSrc & GID::getSourcesMask(configcontext.options().get<std::string>("data-sources"));

  WorkflowSpec specs;
  specs.emplace_back(o2::filtering::getDataFilteringSpec(src, enableSV, useMC));

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, src, src, src, useMC, src);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC);
  if (enableSV) {
    o2::globaltracking::InputHelper::addInputSpecsSVertex(configcontext, specs);
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
