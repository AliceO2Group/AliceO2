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
#include "Framework/Logger.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "TPCWorkflow/RecoWorkflow.h"
#include "TPCWorkflow/PublisherSpec.h"
#include "TPCWorkflow/ClustererSpec.h"
#include "TPCWorkflow/ClusterDecoderRawSpec.h"
#include "TPCWorkflow/CATrackerSpec.h"
#include "TPCWorkflow/EntropyEncoderSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "TPCBase/Digit.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <stdexcept>
#include <algorithm> // std::find
#include <tuple>     // make_tuple

namespace o2
{
namespace tpc
{
namespace reco_workflow
{

using namespace framework;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

const std::unordered_map<std::string, InputType> InputMap{
  {"digitizer", InputType::Digitizer},
  {"digits", InputType::Digits},
  {"clustershardware", InputType::ClustersHardware},
  {"clusters", InputType::Clusters},
  {"zsraw", InputType::ZSRaw},
  {"compressed-clusters", InputType::CompClusters},
  {"encoded-clusters", InputType::EncodedClusters},
};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"digits", OutputType::Digits},
  {"clustershardware", OutputType::ClustersHardware},
  {"clusters", OutputType::Clusters},
  {"tracks", OutputType::Tracks},
  {"compressed-clusters", OutputType::CompClusters},
  {"encoded-clusters", OutputType::EncodedClusters},
  {"disable-writer", OutputType::DisableWriter},
};

framework::WorkflowSpec getWorkflow(std::vector<int> const& tpcSectors, std::vector<int> const& laneConfiguration,
                                    bool propagateMC, unsigned nLanes, std::string const& cfgInput, std::string const& cfgOutput,
                                    int caClusterer)
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

  if (inputType == InputType::ClustersHardware && isEnabled(OutputType::Digits)) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits' from 'clustershardware'");
  }
  if (inputType == InputType::Clusters && (isEnabled(OutputType::Digits) || isEnabled(OutputType::ClustersHardware))) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits', nor 'clustershardware' from 'clusters'");
  }
  if (inputType == InputType::ZSRaw && (isEnabled(OutputType::Clusters) || isEnabled(OutputType::Digits) || isEnabled(OutputType::ClustersHardware))) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits', 'clusters' nor 'clustershardware' from 'zsraw'");
  }

  if (inputType == InputType::ZSRaw && !caClusterer) {
    throw std::invalid_argument("zsraw input needs caclusterer");
  }
  if (caClusterer && (inputType == InputType::Clusters || inputType == InputType::ClustersHardware)) {
    throw std::invalid_argument("ca-clusterer requires digits as input");
  }
  if (caClusterer && (isEnabled(OutputType::Clusters) || isEnabled(OutputType::ClustersHardware))) {
    throw std::invalid_argument("ca-clusterer cannot produce clusters or clustershardware output");
  }

  WorkflowSpec specs;

  // The OutputSpec of the PublisherSpec is configured depending on the input
  // type. Note that the configuration of the dispatch trigger in the main file
  // needs to be done in accordance. This means, if a new input option is added
  // also the dispatch trigger needs to be updated.
  if (inputType == InputType::Digits) {
    using Type = std::vector<o2::tpc::Digit>;
    specs.emplace_back(o2::tpc::getPublisherSpec<Type>(PublisherConf{
                                                         "tpc-digit-reader",
                                                         "o2sim",
                                                         {"digitbranch", "TPCDigit", "Digit branch"},
                                                         {"mcbranch", "TPCDigitMCTruth", "MC label branch"},
                                                         OutputSpec{"TPC", "DIGITS"},
                                                         OutputSpec{"TPC", "DIGITSMCTR"},
                                                         tpcSectors,
                                                         laneConfiguration,
                                                       },
                                                       propagateMC));
  } else if (inputType == InputType::ClustersHardware) {
    specs.emplace_back(o2::tpc::getPublisherSpec(PublisherConf{
                                                   "tpc-clusterhardware-reader",
                                                   "tpcclustershardware",
                                                   {"databranch", "TPCClusterHw", "Branch with TPC ClustersHardware"},
                                                   {"mcbranch", "TPCClusterHwMCTruth", "MC label branch"},
                                                   OutputSpec{"TPC", "CLUSTERHW"},
                                                   OutputSpec{"TPC", "CLUSTERHWMCLBL"},
                                                   tpcSectors,
                                                   laneConfiguration,
                                                 },
                                                 propagateMC));
  } else if (inputType == InputType::Clusters) {
    specs.emplace_back(o2::tpc::getPublisherSpec(PublisherConf{
                                                   "tpc-native-cluster-reader",
                                                   "tpcrec",
                                                   {"clusterbranch", "TPCClusterNative", "Branch with TPC native clusters"},
                                                   {"clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch"},
                                                   OutputSpec{"TPC", "CLUSTERNATIVE"},
                                                   OutputSpec{"TPC", "CLNATIVEMCLBL"},
                                                   tpcSectors,
                                                   laneConfiguration,
                                                 },
                                                 propagateMC));
  } else if (inputType == InputType::CompClusters) {
    // TODO: need to check if we want to store the MC labels alongside with compressed clusters
    // for the moment reading of labels is disabled (last parameter is false)
    // TODO: make a different publisher spec for only one output spec, for now using the
    // PublisherSpec with only sector 0, '_0' is thus appended to the branch name
    specs.emplace_back(o2::tpc::getPublisherSpec(PublisherConf{
                                                   "tpc-compressed-cluster-reader",
                                                   "tpcrec",
                                                   {"clusterbranch", "TPCCompClusters", "Branch with TPC compressed clusters"},
                                                   {"clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch"},
                                                   OutputSpec{"TPC", "COMPCLUSTERS"},
                                                   OutputSpec{"TPC", "CLNATIVEMCLBL"},
                                                   std::vector<int>(1, 0),
                                                   std::vector<int>(1, 0),
                                                 },
                                                 false));
  } else if (inputType == InputType::EncodedClusters) {
    // TODO: need to check if we want to store the MC labels alongside with encoded clusters
    // for the moment reading of labels is disabled (last parameter is false)
    specs.emplace_back(o2::tpc::getPublisherSpec(PublisherConf{
                                                   "tpc-encoded-cluster-reader",
                                                   "tpcrec",
                                                   {"clusterbranch", "TPCEncodedClusters", "Branch with TPC encoded clusters"},
                                                   {"clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch"},
                                                   OutputSpec{"TPC", "ENCCLUSTERS"},
                                                   OutputSpec{"TPC", "CLNATIVEMCLBL"},
                                                   std::vector<int>(1, 0),
                                                   std::vector<int>(1, 0),
                                                 },
                                                 false));
  }

  // output matrix
  // Note: the ClusterHardware format is probably a deprecated legacy format and also the
  // ClusterDecoderRawSpec
  bool produceCompClusters = isEnabled(OutputType::CompClusters) || isEnabled(OutputType::EncodedClusters);
  bool produceTracks = isEnabled(OutputType::Tracks);
  bool runTracker = produceTracks || produceCompClusters;
  bool runHWDecoder = !caClusterer && (runTracker || isEnabled(OutputType::Clusters));
  bool runClusterer = !caClusterer && (runHWDecoder || isEnabled(OutputType::ClustersHardware));
  bool zsDecoder = inputType == InputType::ZSRaw;
  bool runClusterEncoder = isEnabled(OutputType::EncodedClusters);

  // input matrix
  runClusterer &= inputType == InputType::Digitizer || inputType == InputType::Digits;
  runHWDecoder &= runClusterer || inputType == InputType::ClustersHardware;
  runTracker &= caClusterer || (runHWDecoder || inputType == InputType::Clusters);

  WorkflowSpec parallelProcessors;
  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // clusterer process(es)
  //
  //
  if (runClusterer) {
    parallelProcessors.push_back(o2::tpc::getClustererSpec(propagateMC));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // cluster decoder process(es)
  //
  //
  if (runHWDecoder) {
    parallelProcessors.push_back(o2::tpc::getClusterDecoderRawSpec(propagateMC));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // set up parallel TPC lanes
  //
  // the parallelPipeline helper distributes the subspec ids from the lane configuration
  // among the pipelines. All inputs and outputs of processors of one pipeline will be
  // cloned by the number of subspecs served by this pipeline and amended with the subspecs
  parallelProcessors = parallelPipeline(
    parallelProcessors, nLanes,
    [&laneConfiguration]() { return laneConfiguration.size(); },
    [&laneConfiguration](size_t index) { return laneConfiguration[index]; });
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
    auto const* tpcSectorHeader = o2::framework::DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
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
  auto checkReady = [](o2::framework::DataRef const& ref) {
    auto const* tpcSectorHeader = o2::framework::DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
    // sector number -1 indicates end-of-data
    if (tpcSectorHeader != nullptr) {
      if (tpcSectorHeader->sector == -1) {
        // indicate normal processing if not ready and skip if ready
        return std::make_tuple(MakeRootTreeWriterSpec::TerminationCondition::Action::SkipProcessing, true);
      }
    }
    return std::make_tuple(MakeRootTreeWriterSpec::TerminationCondition::Action::DoProcessing, false);
  };

  // -------------------------------------------------------------------------------------------
  // helper to create writer specs for different types of output
  auto makeWriterSpec = [tpcSectors, laneConfiguration, propagateMC, getIndex, getName, checkReady](const char* processName,
                                                                                                    const char* defaultFileName,
                                                                                                    const char* defaultTreeName,
                                                                                                    auto&& databranch,
                                                                                                    auto&& mcbranch) {
    if (tpcSectors.size() == 0) {
      throw std::invalid_argument(std::string("writer process configuration needs list of TPC sectors"));
    }

    auto amendInput = [tpcSectors, laneConfiguration](InputSpec& input, size_t index) {
      input.binding += std::to_string(laneConfiguration[index]);
      DataSpecUtils::updateMatchingSubspec(input, laneConfiguration[index]);
    };
    auto amendBranchDef = [laneConfiguration, propagateMC, amendInput, tpcSectors, getIndex, getName](auto&& def) {
      def.keys = mergeInputs(def.keys, laneConfiguration.size(), amendInput);
      def.nofBranches = tpcSectors.size();
      def.getIndex = getIndex;
      def.getName = getName;
      return std::move(def);
    };

    // depending on the MC propagation flag, the RootTreeWriter spec is created with two
    // or one branch definition
    if (propagateMC) {
      return std::move(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                              MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                              std::move(amendBranchDef(databranch)),
                                              std::move(amendBranchDef(mcbranch)))());
    }
    return std::move(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                            MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                            std::move(amendBranchDef(databranch)))());
  };

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for digits
  //
  // selected by output type 'difits'
  if (isEnabled(OutputType::Digits) && !isEnabled(OutputType::DisableWriter)) {
    using DigitOutputType = std::vector<o2::tpc::Digit>;
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    specs.push_back(makeWriterSpec("tpc-digits-writer",
                                   inputType == InputType::Digits ? "tpc-filtered-digits.root" : "tpcdigits.root",
                                   "o2sim",
                                   BranchDefinition<DigitOutputType>{InputSpec{"data", "TPC", "DIGITS", 0},
                                                                     "TPCDigit",
                                                                     "digit-branch-name"},
                                   BranchDefinition<MCLabelContainer>{InputSpec{"mc", "TPC", "DIGITSMCTR", 0},
                                                                      "TPCDigitMCTruth",
                                                                      "digitmc-branch-name"}));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for hardware clusters
  //
  // selected by output type 'clustershardware'
  if (isEnabled(OutputType::ClustersHardware) && !isEnabled(OutputType::DisableWriter)) {
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    specs.push_back(makeWriterSpec("tpc-clusterhardware-writer",
                                   inputType == InputType::ClustersHardware ? "tpc-filtered-clustershardware.root" : "tpc-clustershardware.root",
                                   "tpcclustershardware",
                                   BranchDefinition<const char*>{InputSpec{"data", "TPC", "CLUSTERHW", 0},
                                                                 "TPCClusterHw",
                                                                 "databranch"},
                                   BranchDefinition<MCLabelContainer>{InputSpec{"mc", "TPC", "CLUSTERHWMCLBL", 0},
                                                                      "TPCClusterHwMCTruth",
                                                                      "mcbranch"}));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for TPC native clusters
  //
  // selected by output type 'clusters'
  if (isEnabled(OutputType::Clusters) && !isEnabled(OutputType::DisableWriter)) {
    using MCLabelCollection = std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>;
    specs.push_back(makeWriterSpec("tpc-native-cluster-writer",
                                   inputType == InputType::Clusters ? "tpc-filtered-native-clusters.root" : "tpc-native-clusters.root",
                                   "tpcrec",
                                   BranchDefinition<const char*>{InputSpec{"data", "TPC", "CLUSTERNATIVE", 0},
                                                                 "TPCClusterNative",
                                                                 "databranch"},
                                   BranchDefinition<MCLabelCollection>{InputSpec{"mc", "TPC", "CLNATIVEMCLBL", 0},
                                                                       "TPCClusterNativeMCTruth",
                                                                       "mcbranch"}));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // tracker process
  //
  // selected by output type 'tracks'
  if (runTracker) {
    specs.emplace_back(o2::tpc::getCATrackerSpec(ca::Config{
                                                   propagateMC ? ca::Operation::ProcessMC : ca::Operation::Noop,
                                                   caClusterer ? ca::Operation::CAClusterer : ca::Operation::Noop,
                                                   zsDecoder ? ca::Operation::ZSDecoder : ca::Operation::Noop,
                                                   produceTracks ? ca::Operation::OutputTracks : ca::Operation::Noop,
                                                   produceCompClusters ? ca::Operation::OutputCompClusters : ca::Operation::Noop,
                                                 },
                                                 laneConfiguration));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // tracker process
  //
  // selected by output type 'encoded-clusters'
  if (runClusterEncoder) {
    specs.emplace_back(o2::tpc::getEntropyEncoderSpec());
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for tracks
  //
  // selected by output type 'tracks'
  if (produceTracks && !isEnabled(OutputType::DisableWriter)) {
    // defining the track writer process using the generic RootTreeWriter and generator tool
    //
    // defaults
    const char* processName = "tpc-track-writer";
    const char* defaultFileName = "tpctracks.root";
    const char* defaultTreeName = "tpcrec";

    //branch definitions for RootTreeWriter spec
    using TrackOutputType = std::vector<o2::tpc::TrackTPC>;

    using ClusRefsOutputType = std::vector<o2::tpc::TPCClRefElem>;

    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    auto tracksdef = BranchDefinition<TrackOutputType>{InputSpec{"inputTracks", "TPC", "TRACKS"},      //
                                                       "TPCTracks", "track-branch-name"};              //
    auto clrefdef = BranchDefinition<ClusRefsOutputType>{InputSpec{"inputClusRef", "TPC", "CLUSREFS"}, //
                                                         "ClusRefs", "trackclusref-branch-name"};      //
    auto mcdef = BranchDefinition<MCLabelContainer>{InputSpec{"mcinput", "TPC", "TRACKMCLBL"},         //
                                                    "TPCTracksMCTruth", "trackmc-branch-name"};        //

    // depending on the MC propagation flag, the RootTreeWriter spec is created with 3 or 2
    // branch definition
    if (propagateMC) {
      specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,            //
                                             MakeRootTreeWriterSpec::TerminationPolicy::Process,       //
                                             MakeRootTreeWriterSpec::TerminationCondition{checkReady}, //
                                             std::move(tracksdef), std::move(clrefdef),                //
                                             std::move(mcdef))());                                     //
    } else {                                                                                           //
      specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,            //
                                             MakeRootTreeWriterSpec::TerminationPolicy::Process,       //
                                             MakeRootTreeWriterSpec::TerminationCondition{checkReady}, //
                                             std::move(tracksdef), std::move(clrefdef))());            //
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for compressed clusters container
  //
  // selected by output type 'compressed-clusters'
  if (produceCompClusters && !isEnabled(OutputType::DisableWriter)) {
    // defining the track writer process using the generic RootTreeWriter and generator tool
    //
    // defaults
    const char* processName = "tpc-compcluster-writer";
    const char* defaultFileName = "tpc-compclusters.root";
    const char* defaultTreeName = "tpcrec";

    //branch definitions for RootTreeWriter spec
    using CCluSerializedType = ROOTSerialized<CompressedClusters>;
    auto ccldef = BranchDefinition<CCluSerializedType>{InputSpec{"inputCompCl", "TPC", "COMPCLUSTERS"}, //
                                                       "TPCCompClusters_0", "compcluster-branch-name"}; //

    specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,            //
                                           MakeRootTreeWriterSpec::TerminationPolicy::Process,       //
                                           MakeRootTreeWriterSpec::TerminationCondition{checkReady}, //
                                           std::move(ccldef))());                                    //
  }

  return std::move(specs);
}

} // end namespace reco_workflow
} // end namespace tpc
} // end namespace o2
