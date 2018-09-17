// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClustererSpec.cxx
/// @author Matthias Richter
/// @since  2018-03-23
/// @brief  spec definition for a TPC clusterer process

#include "ClustererSpec.h"
#include "Headers/DataHeader.h"
#include "TPCBase/Digit.h"
#include "TPCReconstruction/HwClusterer.h"
#include "DataFormatsTPC/Cluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <FairMQLogger.h>
#include <memory> // for make_shared
#include <vector>

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace TPC
{

using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

/// create a processor spec
/// runs the TPC HwClusterer in a DPL process with digits and mc as input
DataProcessorSpec getClustererSpec(bool sendMC)
{
  auto initFunction = [sendMC](InitContext& ic) {
    auto clusterArray = std::make_shared<std::vector<o2::TPC::Cluster>>();
    auto mctruthArray = std::make_shared<MCLabelContainer>();
    // FIXME: correct sector needs to be set!! Do we need a set of clusterers if we want to process
    // multiple sectors?
    auto clusterer = std::make_shared<o2::TPC::HwClusterer>(clusterArray.get(), 0, mctruthArray.get());

    auto processingFct = [clusterer, clusterArray, mctruthArray, sendMC](ProcessingContext& pc) {
      auto inDigits = pc.inputs().get<const std::vector<o2::TPC::Digit>>("digits");
      auto inMCLabels = pc.inputs().get<const MCLabelContainer*>("mclabels");

      LOG(INFO) << "processing " << inDigits.size() << " digit object(s)";
      clusterArray->clear();
      mctruthArray->clear();
      clusterer->process(inDigits, inMCLabels.get());
      LOG(INFO) << "clusterer produced " << clusterArray->size() << " cluster container";
      pc.outputs().snapshot(OutputRef{ "clusters" }, *clusterArray.get());
      if (sendMC) {
        pc.outputs().snapshot(OutputRef{ "clusterlbl" }, *mctruthArray.get());
      }
    };

    return processingFct;
  };

  auto createOutputSpecs = [](bool makeMcOutput) {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{ { "clusters" }, gDataOriginTPC, "CLUSTERSIM", 0, Lifetime::Timeframe },
    };
    if (makeMcOutput) {
      OutputLabel label{ "clusterlbl" };
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{ "tpc-clusterer",
                            { InputSpec{ "digits", gDataOriginTPC, "DIGIT", 0, Lifetime::Timeframe },
                              InputSpec{ "mclabels", gDataOriginTPC, "DIGITMCLBL", 0, Lifetime::Timeframe } },
                            { createOutputSpecs(sendMC) },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
