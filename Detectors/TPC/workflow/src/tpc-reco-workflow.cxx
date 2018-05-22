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

#include "Framework/runDataProcessing.h" // the main driver
#include "Framework/WorkflowSpec.h"

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
/// FIXME:
/// - need to add propagation of command line parameters to this function, this
///   is a feature request in the DPL
/// - add propagation of MC information
/// - add writing of the CA Tracker output to ROOT file
/// This function is required to be implemented to define the workflow specifications
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec specs;

  // choose whether to start from the digits or from the clusters
  // these are just temporary switches
  bool publishDigits = true;
  bool doDecoding = true;

  if (publishDigits) {
    specs.emplace_back(o2::TPC::getDigitReaderSpec());
    specs.emplace_back(o2::TPC::getClustererSpec());
  } else {
    specs.emplace_back(o2::TPC::getClusterReaderSpec());
  }
  specs.emplace_back(o2::TPC::getClusterConverterSpec());

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

  if (true) {
    specs.emplace_back(o2::TPC::getCATrackerSpec());
    specs.emplace_back(o2::TPC::getRootFileWriterSpec());
  } else {
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
