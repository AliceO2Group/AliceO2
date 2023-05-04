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

/// \file   tpc-calib-gainmap-tracks.cxx
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#include <vector>
#include <string>
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/TPCCalibPadGainTracksSpec.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("calib-tpc-gainmap-tracks",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"debug", VariantType::Bool, false, {"create debug tree"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"publish-after-tfs", VariantType::Int, 0, {"number of time frames after which to force publishing the objects"}},
    {"useLastExtractedMapAsReference", VariantType::Bool, false, {"enabling iterative extraction of the gain map: Using the extracted gain map from the previous iteration to correct the cluster charge"}},
    {"polynomialsFile", VariantType::String, "", {"file containing the polynomials for the track topology correction"}},
    {"disablePolynomialsCCDB", VariantType::Bool, false, {"Do not load the polynomials from the CCDB"}},
    {"require-ctp-lumi", o2::framework::VariantType::Bool, false, {"require CTP lumi for TPC correction scaling"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcpadgaintrackscalibrator_configuration.ini");
  auto requireCTPLumi = config.options().get<bool>("require-ctp-lumi");
  const auto debug = config.options().get<bool>("debug");
  const auto publishAfterTFs = (uint32_t)config.options().get<int>("publish-after-tfs");
  const bool useLastExtractedMapAsReference = config.options().get<bool>("useLastExtractedMapAsReference");
  const std::string polynomialsFile = config.options().get<std::string>("polynomialsFile");
  const auto disablePolynomialsCCDB = config.options().get<bool>("disablePolynomialsCCDB");

  WorkflowSpec workflow{getTPCCalibPadGainTracksSpec(publishAfterTFs, debug, useLastExtractedMapAsReference, polynomialsFile, disablePolynomialsCCDB, requireCTPLumi)};
  return workflow;
}
