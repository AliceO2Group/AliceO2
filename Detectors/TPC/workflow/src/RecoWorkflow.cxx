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
#include "TPCWorkflow/ZSSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"

#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <stdexcept>
#include <algorithm> // std::find
#include <tuple>     // make_tuple
#include <array>
#include <gsl/span>

using namespace o2::dataformats;

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
  {"compressed-clusters-ctf", InputType::CompClustersCTF}};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"digits", OutputType::Digits},
  {"clustershardware", OutputType::ClustersHardware},
  {"clusters", OutputType::Clusters},
  {"tracks", OutputType::Tracks},
  {"compressed-clusters", OutputType::CompClusters},
  {"encoded-clusters", OutputType::EncodedClusters},
  {"disable-writer", OutputType::DisableWriter},
  {"send-clusters-per-sector", OutputType::SendClustersPerSector},
  {"zsraw", OutputType::ZSRaw},
  {"qa", OutputType::QA},
  {"no-shared-cluster-map", OutputType::NoSharedClusterMap}};

framework::WorkflowSpec getWorkflow(CompletionPolicyData* policyData, std::vector<int> const& tpcSectors, std::vector<int> const& laneConfiguration,
                                    bool propagateMC, unsigned nLanes, std::string const& cfgInput, std::string const& cfgOutput,
                                    int caClusterer, int zsOnTheFly, int zs10bit, float zsThreshold)
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

  bool decompressTPC = inputType == InputType::CompClustersCTF || inputType == InputType::CompClusters;
  // Disable not applicable settings depending on TPC input, no need to disable manually
  if (decompressTPC && (isEnabled(OutputType::Clusters) || isEnabled(OutputType::Tracks))) {
    caClusterer = false;
    zsOnTheFly = false;
    propagateMC = false;
  }
  if (inputType == InputType::ZSRaw) {
    caClusterer = true;
    zsOnTheFly = false;
    propagateMC = false;
  }
  if (inputType == InputType::ClustersHardware || inputType == InputType::Clusters) {
    caClusterer = false;
  }
  if (!caClusterer) {
    zsOnTheFly = false;
  }

  if (inputType == InputType::ClustersHardware && isEnabled(OutputType::Digits)) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits' from 'clustershardware'");
  }
  if (inputType == InputType::Clusters && (isEnabled(OutputType::Digits) || isEnabled(OutputType::ClustersHardware))) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'digits', nor 'clustershardware' from 'clusters'");
  }
  if (inputType == InputType::ZSRaw && isEnabled(OutputType::ClustersHardware)) {
    throw std::invalid_argument("input/output type mismatch, can not produce 'clustershardware' from 'zsraw'");
  }
  if (caClusterer && (inputType == InputType::Clusters || inputType == InputType::ClustersHardware)) {
    throw std::invalid_argument("ca-clusterer requires digits as input");
  }
  if (caClusterer && (isEnabled(OutputType::ClustersHardware))) {
    throw std::invalid_argument("ca-clusterer cannot produce clustershardware output");
  }

  WorkflowSpec specs;

  // We provide a special publishing method for labels which have been stored in a split format and need
  // to be transformed into a contiguous shareable container before publishing. For other branches/types this returns
  // false and the generic RootTreeWriter publishing proceeds
  static Reader::SpecialPublishHook hook{[](std::string_view name, ProcessingContext& context, o2::framework::Output const& output, char* data) -> bool {
    if (TString(name.data()).Contains("TPCDigitMCTruth") || TString(name.data()).Contains("TPCClusterHwMCTruth") || TString(name.data()).Contains("TPCClusterNativeMCTruth")) {
      auto storedlabels = reinterpret_cast<o2::dataformats::IOMCTruthContainerView const*>(data);
      o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> flatlabels;
      storedlabels->copyandflatten(flatlabels);
      //LOG(INFO) << "PUBLISHING CONST LABELS " << flatlabels.getNElements();
      context.outputs().snapshot(output, flatlabels);
      return true;
    }
    return false;
  }};

  // The OutputSpec of the PublisherSpec is configured depending on the input
  // type. Note that the configuration of the dispatch trigger in the main file
  // needs to be done in accordance. This means, if a new input option is added
  // also the dispatch trigger needs to be updated.
  if (inputType == InputType::Digits) {
    using Type = std::vector<o2::tpc::Digit>;

    specs.emplace_back(o2::tpc::getPublisherSpec<Type>(PublisherConf{
                                                         "tpc-digit-reader",
                                                         "tpcdigits.root",
                                                         "o2sim",
                                                         {"digitbranch", "TPCDigit", "Digit branch"},
                                                         {"mcbranch", "TPCDigitMCTruth", "MC label branch"},
                                                         OutputSpec{"TPC", "DIGITS"},
                                                         OutputSpec{"TPC", "DIGITSMCTR"},
                                                         tpcSectors,
                                                         laneConfiguration,
                                                         &hook},
                                                       propagateMC));
  } else if (inputType == InputType::ClustersHardware) {
    specs.emplace_back(o2::tpc::getPublisherSpec(PublisherConf{
                                                   "tpc-clusterhardware-reader",
                                                   "tpc-clusterhardware.root",
                                                   "tpcclustershardware",
                                                   {"databranch", "TPCClusterHw", "Branch with TPC ClustersHardware"},
                                                   {"mcbranch", "TPCClusterHwMCTruth", "MC label branch"},
                                                   OutputSpec{"TPC", "CLUSTERHW"},
                                                   OutputSpec{"TPC", "CLUSTERHWMCLBL"},
                                                   tpcSectors,
                                                   laneConfiguration,
                                                   &hook},
                                                 propagateMC));
  } else if (inputType == InputType::Clusters) {
    specs.emplace_back(o2::tpc::getPublisherSpec(PublisherConf{
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
                                                 propagateMC));
  } else if (inputType == InputType::CompClusters) {
    // TODO: need to check if we want to store the MC labels alongside with compressed clusters
    // for the moment reading of labels is disabled (last parameter is false)
    // TODO: make a different publisher spec for only one output spec, for now using the
    // PublisherSpec with only sector 0, '_0' is thus appended to the branch name
    specs.emplace_back(o2::tpc::getPublisherSpec(PublisherConf{
                                                   "tpc-compressed-cluster-reader",
                                                   "tpc-compclusters.root",
                                                   "tpcrec",
                                                   {"clusterbranch", "TPCCompClusters", "Branch with TPC compressed clusters"},
                                                   {"", "", ""}, // No MC labels
                                                   OutputSpec{"TPC", "COMPCLUSTERS"},
                                                   OutputSpec{"", ""}, // No MC labels
                                                   std::vector<int>(1, 0),
                                                   std::vector<int>(1, 0),
                                                   &hook},
                                                 false));
  }

  // output matrix
  // Note: the ClusterHardware format is probably a deprecated legacy format and also the
  // ClusterDecoderRawSpec
  bool produceCompClusters = isEnabled(OutputType::CompClusters);
  bool produceTracks = isEnabled(OutputType::Tracks);
  bool runTracker = produceTracks || produceCompClusters || (isEnabled(OutputType::Clusters) && caClusterer);
  bool runHWDecoder = !caClusterer && (runTracker || isEnabled(OutputType::Clusters));
  bool runClusterer = !caClusterer && (runHWDecoder || isEnabled(OutputType::ClustersHardware));
  bool zsDecoder = inputType == InputType::ZSRaw;
  bool runClusterEncoder = isEnabled(OutputType::EncodedClusters);

  // input matrix
  runClusterer &= inputType == InputType::Digitizer || inputType == InputType::Digits;
  runHWDecoder &= runClusterer || inputType == InputType::ClustersHardware;
  runTracker &= caClusterer || runHWDecoder || inputType == InputType::Clusters || decompressTPC;

  bool outRaw = inputType == InputType::Digits && isEnabled(OutputType::ZSRaw);
  //bool runZSDecode = inputType == InputType::ZSRaw;
  bool zsToDigit = inputType == InputType::ZSRaw && isEnabled(OutputType::Digits);

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
    if (tpcSectorHeader->sector() < 0) {
      // special data sets, don't write
      return ~(size_t)0;
    }
    size_t index = 0;
    for (auto const& sector : tpcSectors) {
      if (sector == tpcSectorHeader->sector()) {
        return index;
      }
      ++index;
    }
    throw std::runtime_error("sector " + std::to_string(tpcSectorHeader->sector()) + " not configured for writing");
  };
  auto getName = [tpcSectors](std::string base, size_t index) {
    return base + "_" + std::to_string(tpcSectors.at(index));
  };

  // -------------------------------------------------------------------------------------------
  // helper to create writer specs for different types of output
  auto fillLabels = [](TBranch& branch, std::vector<char> const& labelbuffer, DataRef const& /*ref*/) {
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);
    o2::dataformats::IOMCTruthContainerView outputcontainer;
    auto ptr = &outputcontainer;
    auto br = framework::RootTreeWriter::remapBranch(branch, &ptr);
    outputcontainer.adopt(labelbuffer);
    br->Fill();
    br->ResetAddress();
  };

  auto makeWriterSpec = [tpcSectors, laneConfiguration, propagateMC, getIndex, getName](const char* processName,
                                                                                        const char* defaultFileName,
                                                                                        const char* defaultTreeName,
                                                                                        auto&& databranch,
                                                                                        auto&& mcbranch,
                                                                                        bool singleBranch = false) {
    if (tpcSectors.size() == 0) {
      throw std::invalid_argument(std::string("writer process configuration needs list of TPC sectors"));
    }

    auto amendInput = [tpcSectors, laneConfiguration](InputSpec& input, size_t index) {
      input.binding += std::to_string(laneConfiguration[index]);
      DataSpecUtils::updateMatchingSubspec(input, laneConfiguration[index]);
    };
    auto amendBranchDef = [laneConfiguration, amendInput, tpcSectors, getIndex, getName, singleBranch](auto&& def, bool enableMC = true) {
      if (!singleBranch) {
        def.keys = mergeInputs(def.keys, laneConfiguration.size(), amendInput);
        // the branch is disabled if set to 0
        def.nofBranches = enableMC ? tpcSectors.size() : 0;
        def.getIndex = getIndex;
        def.getName = getName;
      } else {
        // instead of the separate sector branches only one is going to be written
        def.nofBranches = enableMC ? 1 : 0;
      }
      return std::move(def);
    };

    return std::move(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                            std::move(amendBranchDef(databranch)),
                                            std::move(amendBranchDef(mcbranch, propagateMC)))());
  };

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for digits
  //
  // selected by output type 'difits'
  if (isEnabled(OutputType::Digits) && !isEnabled(OutputType::DisableWriter)) {
    using DigitOutputType = std::vector<o2::tpc::Digit>;
    specs.push_back(makeWriterSpec("tpc-digits-writer",
                                   inputType == InputType::ZSRaw ? "tpc-zs-digits.root" : inputType == InputType::Digits ? "tpc-filtered-digits.root" : "tpcdigits.root",
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
    // if the caClusterer is enabled, only one data set with the full TPC is produced, and the writer
    // is configured to write one single branch
    specs.push_back(makeWriterSpec("tpc-native-cluster-writer",
                                   inputType == InputType::Clusters ? "tpc-filtered-native-clusters.root" : "tpc-native-clusters.root",
                                   "tpcrec",
                                   BranchDefinition<const char*>{InputSpec{"data", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}},
                                                                 "TPCClusterNative",
                                                                 "databranch"},
                                   BranchDefinition<std::vector<char>>{InputSpec{"mc", ConcreteDataTypeMatcher{"TPC", "CLNATIVEMCLBL"}},
                                                                       "TPCClusterNativeMCTruth",
                                                                       "mcbranch", fillLabels},
                                   (caClusterer || decompressTPC) && !isEnabled(OutputType::SendClustersPerSector)));
  }

  if (zsOnTheFly) {
    specs.emplace_back(o2::tpc::getZSEncoderSpec(tpcSectors, zs10bit, zsThreshold, outRaw));
  }

  if (zsToDigit) {
    specs.emplace_back(o2::tpc::getZStoDigitsSpec(tpcSectors));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // tracker process
  //
  // selected by output type 'tracks'
  if (runTracker) {
    specs.emplace_back(o2::tpc::getCATrackerSpec(policyData, ca::Config{
                                                               propagateMC ? ca::Operation::ProcessMC : ca::Operation::Noop,
                                                               decompressTPC ? ca::Operation::DecompressTPC : ca::Operation::Noop,
                                                               decompressTPC && inputType == InputType::CompClusters ? ca::Operation::DecompressTPCFromROOT : ca::Operation::Noop,
                                                               caClusterer ? ca::Operation::CAClusterer : ca::Operation::Noop,
                                                               zsDecoder ? ca::Operation::ZSDecoder : ca::Operation::Noop,
                                                               zsOnTheFly ? ca::Operation::ZSOnTheFly : ca::Operation::Noop,
                                                               produceTracks ? ca::Operation::OutputTracks : ca::Operation::Noop,
                                                               produceCompClusters ? ca::Operation::OutputCompClusters : ca::Operation::Noop,
                                                               runClusterEncoder ? ca::Operation::OutputCompClustersFlat : ca::Operation::Noop,
                                                               isEnabled(OutputType::SendClustersPerSector) ? ca::Operation::SendClustersPerSector : ca::Operation::Noop,
                                                               isEnabled(OutputType::QA) ? ca::Operation::OutputQA : ca::Operation::Noop,
                                                               isEnabled(OutputType::Clusters) && (caClusterer || decompressTPC) ? ca::Operation::OutputCAClusters : ca::Operation::Noop,
                                                               isEnabled(OutputType::Clusters) && isEnabled(OutputType::Tracks) && !isEnabled(OutputType::NoSharedClusterMap) ? ca::Operation::OutputSharedClusterMap : ca::Operation::Noop,
                                                             },
                                                 tpcSectors));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // tracker process
  //
  // selected by output type 'encoded-clusters'
  if (runClusterEncoder) {
    specs.emplace_back(o2::tpc::getEntropyEncoderSpec(!runTracker));
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
    // a spectator callback which will be invoked by the tree writer with the extracted object
    // we are using it for printing a log message
    auto logger = BranchDefinition<TrackOutputType>::Spectator([](TrackOutputType const& tracks) {
      LOG(INFO) << "writing " << tracks.size() << " track(s)";
    });
    auto tracksdef = BranchDefinition<TrackOutputType>{InputSpec{"inputTracks", "TPC", "TRACKS", 0},           //
                                                       "TPCTracks", "track-branch-name",                       //
                                                       1,                                                      //
                                                       logger};                                                //
    auto clrefdef = BranchDefinition<ClusRefsOutputType>{InputSpec{"inputClusRef", "TPC", "CLUSREFS", 0},      //
                                                         "ClusRefs", "trackclusref-branch-name"};              //
    auto mcdef = BranchDefinition<std::vector<o2::MCCompLabel>>{InputSpec{"mcinput", "TPC", "TRACKSMCLBL", 0}, //
                                                                "TPCTracksMCTruth",                            //
                                                                (propagateMC ? 1 : 0),                         //
                                                                "trackmc-branch-name"};                        //

    // depending on the MC propagation flag, branch definition for MC labels is disabled
    specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                           std::move(tracksdef), std::move(clrefdef),
                                           std::move(mcdef))());
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
    using CCluSerializedType = ROOTSerialized<CompressedClustersROOT>;
    auto ccldef = BranchDefinition<CCluSerializedType>{InputSpec{"inputCompCl", "TPC", "COMPCLUSTERS"}, //
                                                       "TPCCompClusters_0", "compcluster-branch-name"}; //

    specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,            //
                                           std::move(ccldef))());                                    //
  }

  return std::move(specs);
}

} // end namespace reco_workflow
} // end namespace tpc
} // end namespace o2
