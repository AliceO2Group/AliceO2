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

/// @file   tpc-calibrator-gainmap-tracks.cxx
/// @author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#include <vector>
#include <string>
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/CalibratorPadGainTracksSpec.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"useLastExtractedMapAsReference", VariantType::Bool, false, {"enabling iterative extraction of the gain map: Multiplying the extracted gain map with the latest extracted map stored in the CCDB"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}}};

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
  const bool useLastExtractedMapAsReference = config.options().get<bool>("useLastExtractedMapAsReference");

  WorkflowSpec workflow{getTPCCalibPadGainTracksSpec(useLastExtractedMapAsReference)};
  return workflow;
}
