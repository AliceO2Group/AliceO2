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

/// @file  calib-global-offsets.cxx
#include "FT0Workflow/RecoQCworkflow.h"
#include "Framework/CompletionPolicy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <vector>

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"info-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  GID::mask_t allowedSrc = GID::getSourcesMask("FT0");
  GID::mask_t src = allowedSrc & GID::getSourcesMask(configcontext.options().get<std::string>("info-sources"));

  WorkflowSpec specs;
  specs.emplace_back(o2::ft0::getRecoQCworkflow(src));

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, src, src, src, false, src);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, false);
  o2::globaltracking::InputHelper::addInputSpecsSVertex(configcontext, specs);

  return std::move(specs);
}
