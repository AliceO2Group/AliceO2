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
#include "Utils/MakeRootTreeWriterSpec.h"
#include "TPCWorkflow/RecoWorkflow.h"
#include "DigitReaderSpec.h"
#include "ClusterReaderSpec.h"
#include "ClustererSpec.h"
#include "ClusterConverterSpec.h"
#include "ClusterDecoderRawSpec.h"
#include "CATrackerSpec.h"
#include "RangeTokenizer.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/TrackTPC.h"
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
};

const std::unordered_map<std::string, OutputType> OutputMap{
  { "clusters", OutputType::Clusters },
  { "raw", OutputType::Raw },
  { "decoded-clusters", OutputType::DecodedClusters },
  { "tracks", OutputType::Tracks },
};

framework::WorkflowSpec getWorkflow(bool propagateMC, unsigned nLanes, std::string cfgInput, std::string cfgOutput)
{
  InputType inputType;

  try {
    inputType = InputMap.at(cfgInput);
  } catch (std::out_of_range&) {
    throw std::runtime_error(std::string("invalid input type: ") + cfgInput);
  }
  std::vector<OutputType> outputTypes;
  try {
    outputTypes = RangeTokenizer::tokenize<OutputType>(cfgOutput, [](std::string const& token) { return OutputMap.at(token); });
  } catch (std::out_of_range&) {
    throw std::runtime_error(std::string("invalid output type: ") + cfgOutput);
  }
  auto isEnabled = [&outputTypes](OutputType type) {
    return std::find(outputTypes.begin(), outputTypes.end(), type) != outputTypes.end();
  };

  WorkflowSpec specs;

  // there is no MC info when starting from raw data
  // but maybe the warning can be dropped
  if (propagateMC && inputType == InputType::Raw) {
    LOG(WARNING) << "input type 'raw' selected, switch off MC propagation";
    propagateMC = false;
  }

  // note: converter does not touch MC, this is routed directly to downstream consumer
  if (inputType == InputType::Digits) {
    specs.emplace_back(o2::TPC::getDigitReaderSpec(nLanes));
  } else if (inputType == InputType::Clusters) {
    specs.emplace_back(o2::TPC::getClusterReaderSpec(/*propagateMC*/));
  } else if (inputType == InputType::Raw) {
    throw std::runtime_error(std::string("input type 'raw' not yet implemented"));
  }

  WorkflowSpec parallelProcessors;
  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // clusterer process(es)
  //
  //
  if (inputType == InputType::Digitizer || inputType == InputType::Digits) {
    parallelProcessors.push_back(o2::TPC::getClustererSpec(propagateMC));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // cluster converter process(es)
  //
  //
  if (inputType != InputType::Raw && (isEnabled(OutputType::DecodedClusters) || isEnabled(OutputType::Tracks))) {
    parallelProcessors.push_back(o2::TPC::getClusterConverterSpec(propagateMC));
  }

  if (isEnabled(OutputType::Raw)) {
    throw std::runtime_error(std::string("output types 'clusters' and 'raw' not yet implemented"));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // cluster decoder process(es)
  //
  //
  if (isEnabled(OutputType::DecodedClusters) || isEnabled(OutputType::Tracks)) {
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
                                      input.subSpec = id;
                                    }
                                    for (auto& output : spec.outputs) {
                                      output.subSpec = id;
                                    }
                                  });
  }
  specs.insert(specs.end(), parallelProcessors.begin(), parallelProcessors.end());

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for decoded clusters
  //
  // selected by output type 'decoded-clusters'
  if (isEnabled(OutputType::DecodedClusters)) {
    // writer function
    auto writerFunction = [](InitContext& ic) {
      auto filename = ic.options().get<std::string>("outfile");
      if (filename.empty()) {
        throw std::runtime_error("output file missing");
      }
      auto output = std::make_shared<std::ofstream>(filename.c_str(), std::ios_base::binary);
      return [output](ProcessingContext& pc) {
        LOG(INFO) << "processing data set with " << pc.inputs().size() << " entries";
        for (const auto& entry : pc.inputs()) {
          LOG(INFO) << "  " << *(entry.spec);
          const auto* header = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(entry);
          output->write(reinterpret_cast<const char*>(header), header->headerSize);
          output->write(entry.payload, o2::framework::DataRefUtils::getPayloadSize(entry));
          LOG(INFO) << "wrote data, size " << o2::framework::DataRefUtils::getPayloadSize(entry);
        }
      };
    };

    // inputs to the writer
    auto createInputSpec = [nLanes, propagateMC]() {
      o2::framework::Inputs inputs;
      for (o2::header::DataHeader::SubSpecificationType n = 0; n < nLanes; ++n) {
        inputs.emplace_back(InputSpec{ "input", "TPC", "CLUSTERNATIVE", n, framework::Lifetime::Timeframe });
        if (propagateMC) {
          inputs.emplace_back(InputSpec{ "mcin", "TPC", "CLNATIVEMCLBL", n, framework::Lifetime::Timeframe });
        }
      }
      return std::move(inputs);
    };

    specs.emplace_back(DataProcessorSpec{
      "tpc-decoded-cluster-writer",  //
      { createInputSpec() },         //
      {},                            //
      AlgorithmSpec(writerFunction), //
      Options{
        { "outfile", VariantType::String, "tpc-decoded-clusters.bin", { "Name of the output file" } }, //
      }                                                                                                //
    });
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for tracks
  //
  // selected by output type 'tracks'
  if (isEnabled(OutputType::Tracks)) {
    specs.emplace_back(o2::TPC::getCATrackerSpec(propagateMC, nLanes));

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
      specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName, //
                                             std::move(tracksdef), std::move(mcdef))());    //
    } else {                                                                                //
      specs.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName, //
                                             std::move(tracksdef))());                      //
    }
  }

  return std::move(specs);
}

} // end namespace RecoWorkflow
} // end namespace TPC
} // end namespace o2
