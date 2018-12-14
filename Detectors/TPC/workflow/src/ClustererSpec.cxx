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
#include "Framework/ControlService.h"
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
  std::string processorName = "tpc-clusterer";

  constexpr static size_t NSectors = o2::TPC::Sector::MAXSECTOR;
  struct ProcessAttributes {
    std::vector<o2::TPC::Cluster> clusterArray;
    MCLabelContainer mctruthArray;
    std::array<std::shared_ptr<o2::TPC::HwClusterer>, NSectors> clusterers;
    int verbosity = 1;
    bool finished = false;
  };

  auto initFunction = [sendMC](InitContext& ic) {
    // FIXME: the clusterer needs to be initialized with the sector number, so we need one
    // per sector. Taking a closer look to the HwClusterer, the sector number is only used
    // for calculating the CRU id. This could be achieved by passing the current sector as
    // parameter to the clusterer processing function.
    auto processAttributes = std::make_shared<ProcessAttributes>();

    auto processingFct = [processAttributes, sendMC](ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }
      auto& clusterArray = processAttributes->clusterArray;
      auto& mctruthArray = processAttributes->mctruthArray;
      auto& clusterers = processAttributes->clusterers;
      auto& verbosity = processAttributes->verbosity;
      auto dataref = pc.inputs().get("digits");
      auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(dataref);
      if (sectorHeader == nullptr) {
        LOG(ERROR) << "sector header missing on header stack";
        return;
      }
      auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(dataref);
      o2::header::DataHeader::SubSpecificationType fanSpec = dataHeader->subSpecification;

      const auto sector = sectorHeader->sector;
      if (sector < 0) {
        // forward the control information
        // FIXME define and use flags in TPCSectorHeader
        o2::TPC::TPCSectorHeader header{ sector };
        pc.outputs().snapshot(OutputRef{ "clusters", fanSpec, { header } }, fanSpec);
        if (sendMC) {
          pc.outputs().snapshot(OutputRef{ "clusterlbl", fanSpec, { header } }, fanSpec);
        }
        if (sectorHeader->sector == -1) {
          // got EOD
          processAttributes->finished = true;
          pc.services().get<ControlService>().readyToQuit(false);
        }
        return;
      }
      std::unique_ptr<const MCLabelContainer> inMCLabels;
      if (sendMC) {
        inMCLabels = std::move(pc.inputs().get<const MCLabelContainer*>("mclabels"));
      }
      auto inDigits = pc.inputs().get<const std::vector<o2::TPC::Digit>>("digits");
      if (verbosity > 0 && inMCLabels) {
        LOG(INFO) << "received " << inDigits.size() << " digits, "
                  << inMCLabels->getIndexedSize() << " MC label objects";
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
      // process the digits and MC labels, the bool parameter controls whether to clear all
      // internal data or not. Have to clear it inside the process method as not only the containers
      // are cleared but also the cluster counter. Clearing the containers externally leaves the
      // cluster counter unchanged and leads to an inconsistency between cluster container and
      // MC label container (the latter just grows with every call).
      clusterer->process(inDigits, inMCLabels.get(), true /* clear output containers and cluster counter */);
      const std::vector<o2::TPC::Digit> emptyDigits;
      clusterer->finishProcess(emptyDigits, nullptr, false); // keep here the false, otherwise the clusters are lost of they are not stored in the meantime
      if (verbosity > 0) {
        LOG(INFO) << "clusterer produced " << clusterArray.size() << " cluster(s)";
        if (sendMC) {
          LOG(INFO) << "clusterer produced " << mctruthArray.getIndexedSize() << " MC label object(s)";
        }
      }
      pc.outputs().snapshot(OutputRef{ "clusters", fanSpec, { *sectorHeader } }, clusterArray);
      if (sendMC) {
        pc.outputs().snapshot(OutputRef{ "clusterlbl", fanSpec, { *sectorHeader } }, mctruthArray);
      }
    };

    return processingFct;
  };

  auto createInputSpecs = [](bool makeMcInput) {
    std::vector<InputSpec> inputSpecs{
      InputSpec{ "digits", gDataOriginTPC, "DIGITS", 0, Lifetime::Timeframe },
    };
    if (makeMcInput) {
      constexpr o2::header::DataDescription datadesc("DIGITSMCTR");
      inputSpecs.emplace_back("mclabels", gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(inputSpecs);
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

  return DataProcessorSpec{ processorName,
                            { createInputSpecs(sendMC) },
                            { createOutputSpecs(sendMC) },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
