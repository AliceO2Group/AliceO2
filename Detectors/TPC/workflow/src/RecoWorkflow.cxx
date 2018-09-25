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

#include "TPCWorkflow/RecoWorkflow.h"
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

namespace o2
{
namespace TPC
{
namespace RecoWorkflow
{

using namespace framework;

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

framework::WorkflowSpec getWorkflow(bool propagateMC, int nLanes, std::string cfgInput, std::string cfgOutput)
{
  InputType inputType;
  OutputType outputType;
  try {
    inputType = InputMap.at(cfgInput);
  } catch (std::out_of_range&) {
    throw std::runtime_error(std::string("invalid input type: ") + cfgInput);
  }
  try {
    outputType = OutputMap.at(cfgOutput);
  } catch (std::out_of_range&) {
    throw std::runtime_error(std::string("invalid output type: ") + cfgOutput);
  }

  WorkflowSpec specs;

  // there is no MC info when starting from raw data
  // but maybe the warning can be dropped
  if (propagateMC && outputType == OutputType::Raw) {
    LOG(WARNING) << "input type 'raw' selected, switch off MC propagation";
    propagateMC = false;
  }

  // note: converter does not touch MC, this is routed directly to downstream consumer
  if (inputType == InputType::Digits) {
    specs.emplace_back(o2::TPC::getDigitReaderSpec(nLanes));
    for (int n = 0; n < nLanes; ++n) {
      specs.emplace_back(o2::TPC::getClustererSpec(propagateMC, (nLanes > 1 ? n : -1)));
      specs.emplace_back(o2::TPC::getClusterConverterSpec(propagateMC, (nLanes > 1 ? n : -1)));
    }
  } else if (inputType == InputType::Digitizer) {
    for (int n = 0; n < nLanes; ++n) {
      specs.emplace_back(o2::TPC::getClustererSpec(propagateMC, (nLanes > 1 ? n : -1)));
      specs.emplace_back(o2::TPC::getClusterConverterSpec(propagateMC, (nLanes > 1 ? n : -1)));
    }
  } else if (inputType == InputType::Clusters) {
    specs.emplace_back(o2::TPC::getClusterReaderSpec(/*propagateMC*/));
    for (int n = 0; n < nLanes; ++n) {
      specs.emplace_back(o2::TPC::getClusterConverterSpec(propagateMC, (nLanes > 1 ? n : -1)));
    }
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

  for (int n = 0; n < nLanes; ++n) {
    specs.emplace_back(o2::TPC::getClusterDecoderRawSpec(propagateMC, (nLanes > 1 ? n : -1)));
  }

  auto createInputSpec = []() {
    o2::framework::Inputs inputs;
    /**
    for (uint8_t sector = 0; sector < o2::TPC::Constants::MAXSECTOR; sector++) {
      std::stringstream label;
      label << "input_" << std::setw(2) << std::setfill('0') << (int)sector;
      auto subSpec = o2::TPC::ClusterGroupAttribute{sector, 0}.getSubSpecification();
      inputs.emplace_back(InputSpec{ label.str().c_str(), "TPC", "CLUSTERNATIVE", subSpec, framework::Lifetime::Timeframe });
    }
    */
    inputs.emplace_back(InputSpec{ "input", "TPC", "CLUSTERNATIVE", 0, framework::Lifetime::Timeframe });

    return std::move(inputs);
  };

  if (outputType == OutputType::Tracks) {
    specs.emplace_back(o2::TPC::getCATrackerSpec(propagateMC, nLanes));
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
  return std::move(specs);
}

} // end namespace RecoWorkflow
} // end namespace TPC
} // end namespace o2
