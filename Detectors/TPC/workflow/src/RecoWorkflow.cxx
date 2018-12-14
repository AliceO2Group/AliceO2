// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecoWorkflow.cxx
/// @author Matthias Richter
/// @since  2018-09-26
/// @brief  Workflow definition for the TPC reconstruction

#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Utils/MakeRootTreeWriterSpec.h"
#include "TPCWorkflow/RecoWorkflow.h"
#include "PublisherSpec.h"
#include "ClustererSpec.h"
#include "ClusterConverterSpec.h"
#include "ClusterDecoderRawSpec.h"
#include "CATrackerSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "TPCBase/Digit.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "FairMQLogger.h"
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <stdexcept>
#include <algorithm> // std::find

namespace o2
{
namespace TPC
{
namespace RecoWorkflow
{

using namespace framework;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

const std::unordered_map<std::string, InputType> InputMap{
  { "digitizer", InputType::Digitizer },
  { "digits", InputType::Digits },
  { "clusters", InputType::Clusters },
  { "raw", InputType::Raw },
  { "decoded-clusters", InputType::DecodedClusters },
};

const std::unordered_map<std::string, OutputType> OutputMap{
  { "digits", OutputType::Digits },
  { "clusters", OutputType::Clusters },
  { "raw", OutputType::Raw },
  { "decoded-clusters", OutputType::DecodedClusters },
  { "tracks", OutputType::Tracks },
};

framework::WorkflowSpec getWorkflow(std::vector<int> const& tpcSectors, bool propagateMC, unsigned nLanes, std::string const& cfgInput, std::string const& cfgOutput)
{
  InputType inputType;

  try {
    inputType = InputMap.at(cfgInput);
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid input type: ") + cfgInput);
  }
  std::vector<OutputType> outputTypes;
  try {
    outputTypes = RangeTokenizer::tokenize<OutputType>(cfgOutput, [](std::string const& token) { return OutputMap.at(token); });
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid output type: ") + cfgOutput);
  }
  auto isEnabled = [&outputTypes](OutputType type) {
    return std::find(outputTypes.begin(), outputTypes.end(), type) != outputTypes.end();
  };

  if (inputType == InputType::Clusters && (isEnabled(OutputType::Digits))) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits' from 'clusters");
  }
  if (inputType == InputType::Raw && (isEnabled(OutputType::Digits) || isEnabled(OutputType::Clusters))) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits' nor 'clusters' from 'raw'");
  }
  if (inputType == InputType::DecodedClusters && (isEnabled(OutputType::Clusters) || isEnabled(OutputType::Clusters) || isEnabled(OutputType::Raw))) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits', 'clusters' nor 'raw' from 'decoded-clusters");
  }

  WorkflowSpec specs;

  if (inputType == InputType::Digits) {
    specs.emplace_back(o2::TPC::getPublisherSpec(PublisherConf{
                                                   "tpc-digit-reader",
                                                   "o2sim",
                                                   { "digitbranch", "TPCDigit", "Digit branch" },
                                                   { "mcbranch", "TPCDigitMCTruth", "MC label branch" },
                                                   OutputSpec{ "TPC", "DIGITS" },
                                                   OutputSpec{ "TPC", "DIGITSMCTR" },
                                                   tpcSectors,
                                                   nLanes,
                                                 },
                                                 propagateMC));
  } else if (inputType == InputType::Clusters) {
    specs.emplace_back(o2::TPC::getPublisherSpec(PublisherConf{
                                                   "tpc-cluster-reader",
                                                   "o2sim",
                                                   { "clusterbranch", "TPCCluster", "Cluster branch" },
                                                   { "clustermcbranch", "TPCClusterMCTruth", "MC label branch" },
                                                   OutputSpec{ "TPC", "CLUSTERSIM" },
                                                   OutputSpec{ "TPC", "CLUSTERMCLBL" },
                                                   tpcSectors,
                                                   nLanes,
                                                 },
                                                 propagateMC));
  } else if (inputType == InputType::Raw) {
    specs.emplace_back(o2::TPC::getPublisherSpec(PublisherConf{
                                                   "tpc-raw-cluster-reader",
                                                   "tpcraw",
                                                   { "databranch", "TPCClusterHw", "Branch with raw clusters" },
                                                   { "mcbranch", "TPCClusterHwMCTruth", "MC label branch" },
                                                   OutputSpec{ "TPC", "CLUSTERHW" },
                                                   OutputSpec{ "TPC", "CLUSTERHWMCLBL" },
                                                   tpcSectors,
                                                   nLanes,
                                                 },
                                                 propagateMC));
  } else if (inputType == InputType::DecodedClusters) {
    specs.emplace_back(o2::TPC::getPublisherSpec(PublisherConf{
                                                   "tpc-decoded-cluster-reader",
                                                   "tpcrec",
                                                   { "clusterbranch", "TPCClusterNative", "Branch with decoded clusters" },
                                                   { "clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch" },
                                                   OutputSpec{ "TPC", "CLUSTERNATIVE" },
                                                   OutputSpec{ "TPC", "CLNATIVEMCLBL" },
                                                   tpcSectors,
                                                   nLanes,
                                                 },
                                                 propagateMC));
  }

  // output matrix
  bool runTracker = isEnabled(OutputType::Tracks);
  bool runDecoder = runTracker || isEnabled(OutputType::DecodedClusters);
  bool runConverter = runDecoder || isEnabled(OutputType::Raw);
  bool runClusterer = runConverter || isEnabled(OutputType::Clusters);

  // input matrix
  runClusterer &= inputType == InputType::Digitizer || inputType == InputType::Digits;
  runConverter &= runClusterer || inputType == InputType::Clusters;
  runDecoder &= runConverter || inputType == InputType::Raw;
  runTracker &= runDecoder || inputType == InputType::DecodedClusters;

  WorkflowSpec parallelProcessors;
  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // clusterer process(es)
  //
  //
  if (runClusterer) {
    parallelProcessors.push_back(o2::TPC::getClustererSpec(propagateMC));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // cluster converter process(es)
  //
  //
  if (runConverter) {
    parallelProcessors.push_back(o2::TPC::getClusterConverterSpec(propagateMC));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // cluster decoder process(es)
  //
  //
  if (runDecoder) {
    parallelProcessors.push_back(o2::TPC::getClusterDecoderRawSpec(propagateMC));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // set up parallel TPC lanes
  //
  if (nLanes > 1) {
    parallelProcessors = parallel(parallelProcessors,
                                  nLanes,
                                  [](DataProcessorSpec& spec, size_t id) {
                                    for (auto& input : spec.inputs) {
                                      DataSpecUtils::updateMatchingSubspec(input, id);
                                    }
                                    for (auto& output : spec.outputs) {
                                      output.subSpec = id;
                                    }
                                  });
  }
  specs.insert(specs.end(), parallelProcessors.begin(), parallelProcessors.end());

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // generation of processor specs for various types of outputs
  // based on generic RootTreeWriter and MakeRootTreeWriterSpec generator
  //
  // -------------------------------------------------------------------------------------------
  // the callbacks for the RootTreeWriter
  //
  // The generic writer needs a way to associate incoming data with the individual branches for
  // the TPC sectors. The sector number is transmitted as part of the sector header, the callback
  // finds the corresponding index in the vector of configured sectors
  auto getIndex = [tpcSectors](o2::framework::DataRef const& ref) {
    auto const* tpcSectorHeader = o2::framework::DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(ref);
    if (!tpcSectorHeader) {
      throw std::runtime_error("TPC sector header missing in header stack");
    }
    if (tpcSectorHeader->sector < 0) {
      // special data sets, don't write
      return ~(size_t)0;
    }
    size_t index = 0;
    for (auto const& sector : tpcSectors) {
      if (sector == tpcSectorHeader->sector) {
        return index;
      }
      ++index;
    }
    throw std::runtime_error("sector " + std::to_string(tpcSectorHeader->sector) + " not configured for writing");
  };
  auto getName = [tpcSectors](std::string base, size_t index) {
    return base + "_" + std::to_string(tpcSectors.at(index));
  };

  // check if the process is ready to quit
  // this is decided upon the meta information in the TPC sector header, the operation is set as
  // a negative number in the sector member, -2 indicates no-operation, -1 indicates end-of-data
  // see also PublisherSpec.cxx
  // in this workflow, the EOD is sent after the last real data, and all inputs will receive EOD,
  // so it is enough to check on the first occurence
  // FIXME: this will be changed once DPL can propagate control events like EOD
  auto checkReady = [](o2::framework::DataRef const& ref, bool& isReady) {
    auto const* tpcSectorHeader = o2::framework::DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(ref);
    // sector number -1 indicates end-of-data
    if (tpcSectorHeader != nullptr) {
      isReady = tpcSectorHeader->sector == -1;
      // indicate normal processing if not ready and skip if ready
      if (isReady) {
        return MakeRootTreeWriterSpec::TerminationCondition::Action::SkipProcessing;
      }
    }
    return MakeRootTreeWriterSpec::TerminationCondition::Action::DoProcessing;
  };

  // -------------------------------------------------------------------------------------------
  // helper to create writer specs for different types of output
  auto makeWriterSpec = [tpcSectors, nLanes, propagateMC, getIndex, getName, checkReady](const char* processName,
                                                                                         const char* defaultFileName,
                                                                                         const char* defaultTreeName,
                                                                                         auto&& databranch,
                                                                                         auto&& mcbranch) {
    if (tpcSectors.size() == 0) {
      throw std::invalid_argument(std::string("writer process configuration needs list of TPC sectors"));
    }

    auto amendInput = [](InputSpec& input, size_t lane) {
      input.binding += std::to_string(lane);
      DataSpecUtils::updateMatchingSubspec(input, lane);
    };
    auto amendBranchDef = [nLanes, propagateMC, amendInput, tpcSectors, getIndex, getName](auto&& def) {
      def.keys = mergeInputs(def.keys, nLanes, amendInput);
      def.nofBranches = tpcSectors.size();
      def.getIndex = getIndex;
      def.getName = getName;
      return std::move(def);
    };

    // depending on the MC propagation flag, the RootTreeWriter spec is created with two
    // or one branch definition
    if (propagateMC) {
      return std::move(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                              MakeRootTreeWriterSpec::TerminationCondition{ checkReady },
                                              std::move(amendBranchDef(databranch)),
                                              std::move(amendBranchDef(mcbranch)))());
    }
    return std::move(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                            MakeRootTreeWriterSpec::TerminationCondition{ checkReady },
                                            std::move(amendBranchDef(databranch)))());
  };

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for digits
  //
  // selected by output type 'difits'
  if (isEnabled(OutputType::Digits)) {
    using DigitOutputType = std::vector<o2::TPC::Digit>;
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    specs.push_back(makeWriterSpec("tpc-digits-writer",
                                   inputType == InputType::Digits ? "tpc-filtered-digits.root" : "tpcdigits.root",
                                   "o2sim",
                                   BranchDefinition<DigitOutputType>{ InputSpec{ "data", "TPC", "DIGITS", 0 },
                                                                      "TPCDigit",
                                                                      "digit-branch-name" },
                                   BranchDefinition<MCLabelContainer>{ InputSpec{ "mc", "TPC", "DIGITSMCTR", 0 },
                                                                       "TPCDigitMCTruth",
                                                                       "digitmc-branch-name" }));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for simulated clusters
  //
  // selected by output type 'clusters'
  if (isEnabled(OutputType::Clusters)) {
    using ClusterOutputType = std::vector<o2::TPC::Cluster>;
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    specs.push_back(makeWriterSpec("tpc-cluster-writer",
                                   inputType == InputType::Clusters ? "tpc-filtered-clusters.root" : "tpcclusters.root",
                                   "o2sim",
                                   BranchDefinition<ClusterOutputType>{ InputSpec{ "data", "TPC", "CLUSTERSIM", 0 },
                                                                        "TPCCluster",
                                                                        "cluster-branch-name" },
                                   BranchDefinition<MCLabelContainer>{ InputSpec{ "mc", "TPC", "CLUSTERMCLBL", 0 },
                                                                       "TPCClusterMCTruth",
                                                                       "clustermc-branch-name" }));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for raw hardware clusters
  //
  // selected by output type 'raw'
  if (isEnabled(OutputType::Raw)) {
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    specs.push_back(makeWriterSpec("tpc-raw-cluster-writer",
                                   inputType == InputType::Raw ? "tpc-filtered-raw-clusters.root" : "tpc-raw-clusters.root",
                                   "tpcraw",
                                   BranchDefinition<const char*>{ InputSpec{ "data", "TPC", "CLUSTERHW", 0 },
                                                                  "TPCClusterHw",
                                                                  "databranch" },
                                   BranchDefinition<MCLabelContainer>{ InputSpec{ "mc", "TPC", "CLUSTERHWMCLBL", 0 },
                                                                       "TPCClusterHwMCTruth",
                                                                       "mcbranch" }));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for decoded clusters
  //
  // selected by output type 'decoded-clusters'
  if (isEnabled(OutputType::DecodedClusters)) {
    using MCLabelCollection = std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>;
    specs.push_back(makeWriterSpec("tpc-decoded-cluster-writer",
                                   inputType == InputType::DecodedClusters ? "tpc-filtered-decoded-clusters.root" : "tpc-decoded-clusters.root",
                                   "tpcrec",
                                   BranchDefinition<const char*>{ InputSpec{ "data", "TPC", "CLUSTERNATIVE", 0 },
                                                                  "TPCClusterNative",
                                                                  "databranch" },
                                   BranchDefinition<MCLabelCollection>{ InputSpec{ "mc", "TPC", "CLNATIVEMCLBL", 0 },
                                                                        "TPCClusterNativeMCTruth",
                                                                        "mcbranch" }));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // tracker process
  //
  // selected by output type 'tracks'
  if (runTracker) {
    specs.emplace_back(o2::TPC::getCATrackerSpec(propagateMC, nLanes));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for tracks
  //
  // selected by output type 'tracks'
  if (isEnabled(OutputType::Tracks)) {
    // defining the track writer process using the generic RootTreeWriter and generator tool
    //
    // defaults
    const char* processName = "tpc-track-writer";
    const char* defaultFileName = "tpctracks.root";
    const char* defaultTreeName = "tpcrec";

    //branch definitions for RootTreeWriter spec
    using TrackOutputType = std::vector<o2::TPC::TrackTPC>;
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    auto tracksdef = BranchDefinition<TrackOutputType>{ InputSpec{ "input", "TPC", "TRACKS" },    //
                                                        "TPCTracks", "track-branch-name" };       //
    auto mcdef = BranchDefinition<MCLabelContainer>{ InputSpec{ "mcinput", "TPC", "TRACKMCLBL" }, //
                                                     "TPCTracksMCTruth", "trackmc-branch-name" }; //

    // depending on the MC propagation flag, the RootTreeWriter spec is created with two
    // or one branch definition
    if (propagateMC) {
      specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,              //
                                             MakeRootTreeWriterSpec::TerminationPolicy::Process,         //
                                             MakeRootTreeWriterSpec::TerminationCondition{ checkReady }, //
                                             std::move(tracksdef), std::move(mcdef))());                 //
    } else {                                                                                             //
      specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,              //
                                             MakeRootTreeWriterSpec::TerminationPolicy::Process,         //
                                             MakeRootTreeWriterSpec::TerminationCondition{ checkReady }, //
                                             std::move(tracksdef))());                                   //
    }
  }

  return std::move(specs);
}

} // end namespace RecoWorkflow
} // end namespace TPC
} // end namespace o2
