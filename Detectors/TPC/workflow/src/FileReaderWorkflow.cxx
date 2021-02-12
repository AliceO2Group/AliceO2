// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FileReaderWorkflow.cxx

#include "TPCWorkflow/PublisherSpec.h"
#include "TPCWorkflow/TrackReaderSpec.h"

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

    // We provide a special publishing method for labels which have been stored in a split format and need
    // to be transformed into a contiguous shareable container before publishing. For other branches/types this returns
    // false and the generic RootTreeWriter publishing proceeds
    static RootTreeReader::SpecialPublishHook hook{[](std::string_view name, ProcessingContext& context, o2::framework::Output const& output, char* data) -> bool {
      if (TString(name.data()).Contains("TPCDigitMCTruth") || TString(name.data()).Contains("TPCClusterHwMCTruth") || TString(name.data()).Contains("TPCClusterNativeMCTruth")) {
        auto storedlabels = reinterpret_cast<o2::dataformats::IOMCTruthContainerView const*>(data);
        o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> flatlabels;
        storedlabels->copyandflatten(flatlabels);
        LOG(INFO) << "PUBLISHING CONST LABELS " << flatlabels.getNElements();
        context.outputs().snapshot(output, flatlabels);
        return true;
      }
      return false;
    }};

    std::vector<int> tpcSectors(36);
    std::iota(tpcSectors.begin(), tpcSectors.end(), 0);
    std::vector<int> laneConfiguration = tpcSectors;

    specs.emplace_back(o2::tpc::getPublisherSpec(o2::tpc::PublisherConf{
                                                   "tpc-native-cluster-reader",
                                                   "tpc-native-clusters.root",
                                                   "tpcrec",
                                                   {"clusterbranch", "TPCClusterNative", "Branch with TPC native clusters"},
                                                   {"clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch"},
                                                   OutputSpec{"TPC", "CLUSTERNATIVE"},
                                                   OutputSpec{"TPC", "CLNATIVEMCLBL"},
                                                   tpcSectors,
                                                   laneConfiguration,
                                                   &hook},
                                                 doMC));
  }

  if (isEnabled(Input::Tracks)) {

    specs.push_back(o2::tpc::getTPCTrackReaderSpec(doMC));
  }

  return std::move(specs);
}
