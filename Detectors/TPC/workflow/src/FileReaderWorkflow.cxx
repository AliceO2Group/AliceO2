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

/// @file   FileReaderWorkflow.cxx

#include "TPCReaderWorkflow/ClusterReaderSpec.h"
#include "TPCReaderWorkflow/TriggerReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"

#include "Algorithm/RangeTokenizer.h"

#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  using namespace o2::framework;

  std::vector<ConfigParamSpec> options{
    {"input-type", VariantType::String, "clusters", {"clusters, tracks"}},
    {"disable-mc", VariantType::Bool, false, {"disable sending of MC information"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

enum struct Input { Clusters,
                    Tracks
};

const std::unordered_map<std::string, Input> InputMap{
  {"clusters", Input::Clusters},
  {"tracks", Input::Tracks}};

/// MC info is processed by default, disabled by using command line option `--disable-mc`
///
/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;

  auto inputType = cfgc.options().get<std::string>("input-type");
  bool doMC = not cfgc.options().get<bool>("disable-mc");

  std::vector<Input> inputTypes;
  try {
    inputTypes = o2::RangeTokenizer::tokenize<Input>(inputType, [](std::string const& token) { return InputMap.at(token); });
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid input type: ") + inputType);
  }
  auto isEnabled = [&inputTypes](Input type) {
    return std::find(inputTypes.begin(), inputTypes.end(), type) != inputTypes.end();
  };

  if (isEnabled(Input::Clusters)) {
    specs.emplace_back(o2::tpc::getClusterReaderSpec(doMC));
    if (!getenv("DPL_DISABLE_TPC_TRIGGER_READER") || atoi(getenv("DPL_DISABLE_TPC_TRIGGER_READER")) != 1) {
      specs.emplace_back(o2::tpc::getTPCTriggerReaderSpec());
    }
  }

  if (isEnabled(Input::Tracks)) {

    specs.push_back(o2::tpc::getTPCTrackReaderSpec(doMC));
  }

  return std::move(specs);
}
