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
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ClusterHardware.h"
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
DataProcessorSpec getClustererSpec()
{
  auto initFunction = [](InitContext& ic) {
    auto clusterArray = std::make_shared<std::vector<o2::TPC::ClusterHardwareContainer8kb>>();
    auto mctruthArray = std::make_shared<MCLabelContainer>();
    auto clusterer = std::make_shared<o2::TPC::HwClusterer>(clusterArray, mctruthArray);

    auto processingFct = [clusterer, clusterArray, mctruthArray](ProcessingContext& pc) {
      auto inDigits = std::make_shared<const std::vector<o2::TPC::Digit>>(
        pc.inputs().get<const std::vector<o2::TPC::Digit>>("digits"));
      auto inMCLabels = std::shared_ptr<const MCLabelContainer>(
        pc.inputs().get<const MCLabelContainer>("mclabels"));

      LOG(INFO) << "processing " << inDigits->size() << " digit object(s)";
      clusterArray->clear();
      mctruthArray->clear();
      clusterer->Process(inDigits, inMCLabels, 1);
      LOG(INFO) << "clusterer produced " << clusterArray->size() << " cluster(s)";
      pc.outputs().snapshot(OutputRef{ "clusters" }, *clusterArray.get());
      pc.outputs().snapshot(OutputRef{ "clusterlbl" }, *mctruthArray.get());
    };

    return processingFct;
  };

  return DataProcessorSpec{ "tpc-clusterer",
                            { InputSpec{ "digits", gDataOriginTPC, "DIGIT", 0, Lifetime::Timeframe },
                              InputSpec{ "mclabels", gDataOriginTPC, "DIGITMCLBL", 0, Lifetime::Timeframe } },
                            { OutputSpec{ { "clusters" }, gDataOriginTPC, "CLUSTERSIM", 0, Lifetime::Timeframe },
                              OutputSpec{ { "clusterlbl" }, gDataOriginTPC, "CLUSTERMCLBL", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
