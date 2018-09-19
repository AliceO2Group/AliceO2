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
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
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
  constexpr static size_t NSectors = o2::TPC::Sector::MAXSECTOR;
  struct ProcessAttributes {
    std::vector<o2::TPC::Cluster> clusterArray;
    MCLabelContainer mctruthArray;
    std::array<std::shared_ptr<o2::TPC::HwClusterer>, NSectors> clusterers;
    int verbosity = 1;
  };

  auto initFunction = [sendMC](InitContext& ic) {
    // FIXME: the clusterer needs to be initialized with the sector number, so we need one
    // per sector. Taking a closer look to the HwClusterer, the sector number is only used
    // for calculating the CRU id. This could be achieved by passing the current sector as
    // parameter to the clusterer processing function.
    auto processAttributes = std::make_shared<ProcessAttributes>();

    auto processingFct = [processAttributes, sendMC](ProcessingContext& pc) {
      auto& clusterArray = processAttributes->clusterArray;
      auto& mctruthArray = processAttributes->mctruthArray;
      auto& clusterers = processAttributes->clusterers;
      auto& verbosity = processAttributes->verbosity;
      auto inDigits = pc.inputs().get<const std::vector<o2::TPC::Digit>>("digits");
      if (verbosity > 0) {
        LOG(INFO) << "received " << inDigits.size() << " digits";
      }
      auto inMCLabels = pc.inputs().get<const MCLabelContainer*>("mclabels");
      auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("mclabels"));
      if (sectorHeader == nullptr) {
        LOG(ERROR) << "sector header missing on header stack";
        return;
      }
      const auto sector = sectorHeader->sector;
      if (sector < 0) {
        // FIXME: we need to propagate something
        return;
      }
      if (!clusterers[sector]) {
        // create the clusterer for this sector, take the same target arrays for all clusterers
        // as they are not invoked in parallel
        // the cost of creating the clusterer should be small so we do it in the processing
        clusterers[sector] = std::make_shared<o2::TPC::HwClusterer>(&clusterArray, sector, &mctruthArray);
      }
      auto& clusterer = clusterers[sector];

      if (verbosity > 0) {
        LOG(INFO) << "processing " << inDigits.size() << " digit object(s) of sector " << sectorHeader->sector;
      }
      clusterArray.clear();
      mctruthArray.clear();
      clusterer->process(inDigits, inMCLabels.get());
      const std::vector<o2::TPC::Digit> emptyDigits;
      clusterer->finishProcess(emptyDigits, nullptr);
      if (verbosity > 0) {
        LOG(INFO) << "clusterer produced " << clusterArray.size() << " cluster container";
      }
      pc.outputs().snapshot(OutputRef{ "clusters", 0, { *sectorHeader } }, clusterArray);
      if (sendMC) {
        pc.outputs().snapshot(OutputRef{ "clusterlbl", 0, { *sectorHeader } }, mctruthArray);
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
