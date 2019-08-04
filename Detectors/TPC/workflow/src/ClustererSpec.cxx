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

#include "TPCWorkflow/ClustererSpec.h"
#include "Framework/ControlService.h"
#include "Headers/DataHeader.h"
#include "TPCBase/Digit.h"
#include "TPCReconstruction/HwClusterer.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <FairMQLogger.h>
#include <memory> // for make_shared
#include <vector>
#include <numeric>   // std::accumulate
#include <algorithm> // std::copy

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace tpc
{

using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

/// create a processor spec
/// runs the TPC HwClusterer in a DPL process with digits and mc as input
DataProcessorSpec getClustererSpec(bool sendMC, bool haveDigTriggers)
{
  std::string processorName = "tpc-clusterer";

  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  struct ProcessAttributes {
    std::vector<o2::tpc::ClusterHardwareContainer8kb> clusterArray;
    MCLabelContainer mctruthArray;
    std::array<std::shared_ptr<o2::tpc::HwClusterer>, NSectors> clusterers;
    int verbosity = 1;
    bool finished = false;
  };

  auto initFunction = [](InitContext& ic) {
    // FIXME: the clusterer needs to be initialized with the sector number, so we need one
    // per sector. Taking a closer look to the HwClusterer, the sector number is only used
    // for calculating the CRU id. This could be achieved by passing the current sector as
    // parameter to the clusterer processing function.
    auto processAttributes = std::make_shared<ProcessAttributes>();

    auto processSectorFunction = [processAttributes](ProcessingContext& pc, std::string inputKey, std::string labelKey) -> bool {
      auto& clusterArray = processAttributes->clusterArray;
      auto& mctruthArray = processAttributes->mctruthArray;
      auto& clusterers = processAttributes->clusterers;
      auto& verbosity = processAttributes->verbosity;
      auto dataref = pc.inputs().get(inputKey);
      auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(dataref);
      if (sectorHeader == nullptr) {
        LOG(ERROR) << "sector header missing on header stack";
        return false;
      }
      auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(dataref);
      o2::header::DataHeader::SubSpecificationType fanSpec = dataHeader->subSpecification;

      const auto sector = sectorHeader->sector;
      if (sector < 0) {
        // forward the control information
        // FIXME define and use flags in TPCSectorHeader
        o2::tpc::TPCSectorHeader header{sector};
        pc.outputs().snapshot(Output{gDataOriginTPC, "CLUSTERHW", fanSpec, Lifetime::Timeframe, {header}}, fanSpec);
        if (!labelKey.empty()) {
          pc.outputs().snapshot(Output{gDataOriginTPC, "CLUSTERHWMCLBL", fanSpec, Lifetime::Timeframe, {header}}, fanSpec);
        }
        return (sectorHeader->sector == -1);
      }
      std::unique_ptr<const MCLabelContainer> inMCLabels;
      if (!labelKey.empty()) {
        inMCLabels = std::move(pc.inputs().get<const MCLabelContainer*>(labelKey.c_str()));
      }
      auto inDigits = pc.inputs().get<const std::vector<o2::tpc::Digit>>(inputKey.c_str());
      if (verbosity > 0 && inMCLabels) {
        LOG(INFO) << "received " << inDigits.size() << " digits, "
                  << inMCLabels->getIndexedSize() << " MC label objects";
      }
      if (!clusterers[sector]) {
        // create the clusterer for this sector, take the same target arrays for all clusterers
        // as they are not invoked in parallel
        // the cost of creating the clusterer should be small so we do it in the processing
        clusterers[sector] = std::make_shared<o2::tpc::HwClusterer>(&clusterArray, sector, &mctruthArray);
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
      const std::vector<o2::tpc::Digit> emptyDigits;
      clusterer->finishProcess(emptyDigits, nullptr, false); // keep here the false, otherwise the clusters are lost of they are not stored in the meantime
      if (verbosity > 0) {
        LOG(INFO) << "clusterer produced "
                  << std::accumulate(clusterArray.begin(), clusterArray.end(), size_t(0), [](size_t l, auto const& r) { return l + r.getContainer()->numberOfClusters; })
                  << " cluster(s)";
        if (!labelKey.empty()) {
          LOG(INFO) << "clusterer produced " << mctruthArray.getIndexedSize() << " MC label object(s)";
        }
      }
      // FIXME: that should be a case for pmr, want to send the content of the vector as a binary
      // block by using move semantics
      auto outputPages = pc.outputs().make<ClusterHardwareContainer8kb>(Output{gDataOriginTPC, "CLUSTERHW", fanSpec, Lifetime::Timeframe, {*sectorHeader}}, clusterArray.size());
      std::copy(clusterArray.begin(), clusterArray.end(), outputPages.begin());
      if (!labelKey.empty()) {
        pc.outputs().snapshot(Output{gDataOriginTPC, "CLUSTERHWMCLBL", fanSpec, Lifetime::Timeframe, {*sectorHeader}}, mctruthArray);
      }
      return false;
    };

    auto processingFct = [processAttributes, processSectorFunction](ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      struct SectorInputDesc {
        std::string inputKey = "";
        std::string labelKey = "";
      };
      std::map<o2::header::DataHeader::SubSpecificationType, SectorInputDesc> inputs;
      for (auto const& inputRef : pc.inputs()) {
        auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
        assert(dataHeader);
        if (dataHeader->dataOrigin == gDataOriginTPC && dataHeader->dataDescription == o2::header::DataDescription("DIGITS")) {
          inputs[dataHeader->subSpecification].inputKey = inputRef.spec->binding;
        } else if (dataHeader->dataOrigin == gDataOriginTPC && dataHeader->dataDescription == o2::header::DataDescription("DIGITSMCTR")) {
          inputs[dataHeader->subSpecification].labelKey = inputRef.spec->binding;
        }
      }
      bool finished = true;
      for (auto const& input : inputs) {
        if (!processSectorFunction(pc, input.second.inputKey, input.second.labelKey)) {
          finished = false;
        }
      }
      if (finished) {
        // got EOD on all inputs
        processAttributes->finished = true;
        pc.services().get<ControlService>().readyToQuit(false);
      }
    };
    return processingFct;
  };

  auto createInputSpecs = [](bool makeMcInput, bool makeTriggersInput = false) {
    std::vector<InputSpec> inputSpecs{
      InputSpec{"digits", gDataOriginTPC, "DIGITS", 0, Lifetime::Timeframe},
    };
    if (makeMcInput) {
      constexpr o2::header::DataDescription datadesc("DIGITSMCTR");
      inputSpecs.emplace_back("mclabels", gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    if (makeTriggersInput) {
      // this is an additional output by the TPC digitizer, need to check if that
      // has to go into the clusterer as well
      constexpr o2::header::DataDescription datadesc("DIGTRIGGERS");
      inputSpecs.emplace_back("digtriggers", gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(inputSpecs);
  };

  auto createOutputSpecs = [](bool makeMcOutput) {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{{"clusters"}, gDataOriginTPC, "CLUSTERHW", 0, Lifetime::Timeframe},
    };
    if (makeMcOutput) {
      OutputLabel label{"clusterlbl"};
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERHWMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{processorName,
                           {createInputSpecs(sendMC, haveDigTriggers)},
                           {createOutputSpecs(sendMC)},
                           AlgorithmSpec(initFunction)};
}

} // namespace tpc
} // namespace o2
