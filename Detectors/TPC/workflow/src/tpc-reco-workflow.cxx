// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tpc-reco-workflow.cxx
/// @author Matthias Richter
/// @since  2018-03-15
/// @brief  Basic DPL workflow for TPC reconstruction starting from digits

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"

#include "DigitReaderSpec.h"
#include "ClusterReaderSpec.h"
#include "ClustererSpec.h"
#include "ClusterConverterSpec.h"
#include "ClusterDecoderRawSpec.h"
#include "CATrackerSpec.h"
#include "RootFileWriterSpec.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"

#include "FairMQLogger.h"
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <stdexcept>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    { "input-type", o2::framework::VariantType::String, "digits", { "digits, clusters, raw" } },
    { "output-type", o2::framework::VariantType::String, "tracks", { "clusters, raw, tracks" } },
    { "disable-mc", o2::framework::VariantType::Bool, false, { "disable sending of MC information" } },
  };
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

/// Defines basic workflow for TPC reconstruction
/// - digit reader
/// - clusterer
/// - cluster converter
/// - cluster raw decoder
/// - CA tracker
///
/// Digit reader and clusterer can be replaced by the cluster reader.
///
/// MC info is always sent by the digit reader and clusterer processes, the
/// cluster converter process creating the raw format can be configured to forward MC.
///
/// FIXME:
/// - add propagation of MC information
/// - add writing of the CA Tracker output to ROOT file
/// This function is required to be implemented to define the workflow specifications
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;

  /// extract the workflow options and configure workflow
  enum struct InputType { Digits,
                          Clusters,
                          Raw };
  enum struct OutputType { Clusters,
                           Raw,
                           DecodedClusters,
                           Tracks };

  const std::unordered_map<std::string, InputType> InputMap{
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

  InputType inputType;
  OutputType outputType;
  bool propagateMC = not cfgc.options().get<bool>("disable-mc");
  try {
    inputType = InputMap.at(cfgc.options().get<std::string>("input-type"));
  } catch (std::out_of_range&) {
    throw std::runtime_error(std::string("invalid input type: ") + cfgc.options().get<std::string>("input-type"));
  }
  try {
    outputType = OutputMap.at(cfgc.options().get<std::string>("output-type"));
  } catch (std::out_of_range&) {
    throw std::runtime_error(std::string("invalid output type: ") + cfgc.options().get<std::string>("output-type"));
  }

  // there is no MC info when starting from raw data
  // but maybe the warning can be dropped
  if (propagateMC && outputType == OutputType::Raw) {
    LOG(WARNING) << "input type 'raw' selected, switch off MC propagation";
    propagateMC = false;
  }

  // note: converter does not touch MC, this is routed directly to downstream consumer
  if (inputType == InputType::Digits) {
    specs.emplace_back(o2::TPC::getDigitReaderSpec());
    specs.emplace_back(o2::TPC::getClustererSpec(propagateMC));
    specs.emplace_back(o2::TPC::getClusterConverterSpec(false));
  } else if (inputType == InputType::Clusters) {
    specs.emplace_back(o2::TPC::getClusterReaderSpec(/*propagateMC*/));
    specs.emplace_back(o2::TPC::getClusterConverterSpec(false));
  } else if (inputType == InputType::Raw) {
    throw std::runtime_error(std::string("input type 'raw' not yet implemented"));
  }

  if (outputType == OutputType::Clusters || outputType == OutputType::Raw) {
    throw std::runtime_error(std::string("output types 'clusters' and 'raw' not yet implemented"));
  }
  // also add a binary writer
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
        output->write(entry.payload, o2::framework::DataRefUtils::getPayloadSize(entry));
        LOG(INFO) << "wrote data, size " << o2::framework::DataRefUtils::getPayloadSize(entry);
      }
    };
  };

  specs.emplace_back(o2::TPC::getClusterDecoderRawSpec());

  auto createInputSpec = []() {
    o2::framework::Inputs inputs;
    /**
    for (uint8_t sector = 0; sector < o2::TPC::Constants::MAXSECTOR; sector++) {
      std::stringstream label;
      label << "input_" << std::setw(2) << std::setfill('0') << (int)sector;
      auto subSpec = o2::TPC::ClusterGroupAttribute{sector, 0}.getSubSpecification();
      inputs.emplace_back(InputSpec{ label.str().c_str(), "TPC", "CLUSTERNATIVE", subSpec, Lifetime::Timeframe });
    }
    */
    inputs.emplace_back(InputSpec{ "input", "TPC", "CLUSTERNATIVE", 0, Lifetime::Timeframe });

    return std::move(inputs);
  };

  if (outputType == OutputType::Tracks) {
    specs.emplace_back(o2::TPC::getCATrackerSpec());
    specs.emplace_back(o2::TPC::getRootFileWriterSpec());
  } else if (outputType == OutputType::DecodedClusters) {
    specs.emplace_back(DataProcessorSpec{ "writer",
                                          { createInputSpec() },
                                          {},
                                          AlgorithmSpec(writerFunction),
                                          Options{
                                            { "outfile", VariantType::String, { "Name of the output file" } },
                                          } });
  }
  return specs;
}
