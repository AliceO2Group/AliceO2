// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CATrackerSpec.cxx
/// @author Matthias Richter
/// @since  2018-04-18
/// @brief  Processor spec for running TPC CA tracking

#include "CATrackerSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/DataRefUtils.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/TPCCATracking.h"
#include "TPCBase/Sector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Algorithm/Parser.h"
#include <FairMQLogger.h>
#include <memory> // for make_shared
#include <vector>
#include <iomanip>

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace TPC
{

using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

DataProcessorSpec getCATrackerSpec(bool processMC, size_t fanIn)
{
  constexpr static size_t NSectors = o2::TPC::Sector::MAXSECTOR;
  using ClusterGroupParser = o2::algorithm::ForwardParser<o2::TPC::ClusterGroupHeader>;
  struct ProcessAttributes {
    // the input comes in individual calls and we need to buffer until
    // data set is complete, have to think about a DPL feature to take
    // ownership of an input
    std::array<std::vector<unsigned char>, NSectors> inputs;
    std::array<std::vector<MCLabelContainer>, NSectors> mcInputs;
    std::bitset<NSectors> validInputs = 0;
    std::bitset<NSectors> validMcInputs = 0;
    std::unique_ptr<ClusterGroupParser> parser;
    std::unique_ptr<o2::TPC::TPCCATracking> tracker;
    int verbosity = 1;
    size_t nParallelInputs = 1;
  };

  auto initFunction = [processMC, fanIn](InitContext& ic) {
    auto options = ic.options().get<std::string>("tracker-options");

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      processAttributes->nParallelInputs = fanIn;
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      parser = std::make_unique<ClusterGroupParser>();
      tracker = std::make_unique<o2::TPC::TPCCATracking>();
      tracker->initialize(options.c_str());
      processAttributes->validInputs.reset();
      processAttributes->validMcInputs.reset();
    }

    auto processingFct = [processAttributes, processMC](ProcessingContext& pc) {
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      uint64_t activeSectors = 0;
      auto& verbosity = processAttributes->verbosity;

      // FIXME cleanup almost duplicated code
      auto& validMcInputs = processAttributes->validMcInputs;
      auto& mcInputs = processAttributes->mcInputs;
      if (processMC) {
        // we can later extend this to multiple inputs
        std::vector<std::string> inputLabels(processAttributes->nParallelInputs);
        std::generate(inputLabels.begin(), inputLabels.end(), [counter = std::make_shared<int>(0)]() { return "mclblin" + std::to_string((*counter)++); });
        for (auto& inputLabel : inputLabels) {
          auto ref = pc.inputs().get(inputLabel);
          auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(ref);
          if (sectorHeader == nullptr) {
            // FIXME: think about error policy
            LOG(ERROR) << "sector header missing on header stack";
            return;
          }
          const int& sector = sectorHeader->sector;
          if (sector < 0) {
            // FIXME: this we have to sort out once the steering is implemented
            continue;
          }
          if (validMcInputs.test(sector)) {
            // have already data for this sector, this should not happen in the current
            // sequential implementation, for parallel path merged at the tracker stage
            // multiple buffers need to be handled
            throw std::runtime_error("can only have one data set per sector");
          }
          mcInputs[sector] = std::move(pc.inputs().get<std::vector<MCLabelContainer>>(inputLabel.c_str()));
          validMcInputs.set(sector);
          activeSectors |= sectorHeader->activeSectors;
          if (verbosity > 1) {
            LOG(INFO) << "received " << *(ref.spec) << " MC label containers"
                      << " for sector " << sector                                      //
                      << std::endl                                                     //
                      << "  mc input status:   " << validMcInputs                      //
                      << std::endl                                                     //
                      << "  active sectors: " << std::bitset<NSectors>(activeSectors); //
          }
        }
      }

      std::vector<std::string> inputLabels(processAttributes->nParallelInputs);
      std::generate(inputLabels.begin(), inputLabels.end(), [counter = std::make_shared<int>(0)]() { return "input" + std::to_string((*counter)++); });
      auto& validInputs = processAttributes->validInputs;
      auto& inputs = processAttributes->inputs;
      for (auto& inputLabel : inputLabels) {
        auto ref = pc.inputs().get(inputLabel);
        auto payploadSize = DataRefUtils::getPayloadSize(ref);
        auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(ref);
        if (sectorHeader == nullptr) {
          // FIXME: think about error policy
          LOG(ERROR) << "sector header missing on header stack";
          return;
        }
        const int& sector = sectorHeader->sector;
        if (sector < 0) {
          // FIXME: this we have to sort out once the steering is implemented
          continue;
        }
        if (validInputs.test(sector)) {
          // have already data for this sector, this should not happen in the current
          // sequential implementation, for parallel path merged at the tracker stage
          // multiple buffers need to be handled
          throw std::runtime_error("can only have one data set per sector");
        }
        inputs[sector].resize(payploadSize);
        std::copy(ref.payload, ref.payload + payploadSize, inputs[sector].begin());
        validInputs.set(sector);
        activeSectors |= sectorHeader->activeSectors;
        if (verbosity > 1) {
          LOG(INFO) << "received " << *(ref.spec) << ", size " << inputs[sector].size() //
                    << " for sector " << sector                                         //
                    << std::endl                                                        //
                    << "  input status:   " << validInputs                              //
                    << std::endl                                                        //
                    << "  active sectors: " << std::bitset<NSectors>(activeSectors);    //
        }
      }

      if (activeSectors == 0 || (activeSectors & validInputs.to_ulong()) != activeSectors ||
          (processMC && (activeSectors & validMcInputs.to_ulong()) != activeSectors)) {
        // not all sectors available
        // FIXME: do we need to send something
        return;
      }
      assert(processMC == false || validMcInputs == validInputs);
      if (verbosity > 0) {
        LOG(INFO) << "running tracking for sectors " << validInputs;
      }
      ClusterNativeAccessFullTPC clusterIndex;
      memset(&clusterIndex, 0, sizeof(clusterIndex));
      for (size_t index = 0; index < NSectors; index++) {
        if (!validInputs.test(index)) {
          continue;
        }
        const auto& input = inputs[index];
        auto mcIterator = mcInputs[index].begin();
        parser->parse(&input.front(), input.size(),
                      [](const typename ClusterGroupParser::HeaderType& h) {
                        // check the header, but in this case there is no validity check
                        return true;
                      },
                      [](const typename ClusterGroupParser::HeaderType& h) {
                        // get the size of the frame including payload
                        // and header and trailer size, e.g. payload size
                        // from a header member
                        return h.nClusters * sizeof(ClusterNative) + ClusterGroupParser::totalOffset;
                      },
                      [&](typename ClusterGroupParser::FrameInfo& frame) {
                        int sector = frame.header->sector;
                        int padrow = frame.header->globalPadRow;
                        int nClusters = frame.header->nClusters;
                        LOG(DEBUG) << "   sector " << std::setw(2) << std::setfill('0') << sector << "   padrow "
                                   << std::setw(3) << std::setfill('0') << padrow << " clusters " << std::setw(3)
                                   << std::setfill(' ') << nClusters;
                        clusterIndex.clusters[sector][padrow] = reinterpret_cast<const ClusterNative*>(frame.payload);
                        clusterIndex.nClusters[sector][padrow] = nClusters;
                        assert(processMC == false || mcIterator != mcInputs[index].end());
                        if (processMC && mcIterator != mcInputs[index].end()) {
                          clusterIndex.clustersMCTruth[sector][padrow] = &(*mcIterator);
                          ++mcIterator;
                        }

                        return true;
                      });
      }

      std::vector<TrackTPC> tracks;
      MCLabelContainer tracksMCTruth;
      int retVal = tracker->runTracking(clusterIndex, &tracks, (processMC ? &tracksMCTruth : nullptr));
      if (retVal != 0) {
        // FIXME: error policy
        LOG(ERROR) << "tracker returned error code " << retVal;
      }
      LOG(INFO) << "found " << tracks.size() << " track(s)";
      pc.outputs().snapshot(OutputRef{ "output" }, tracks);
      if (processMC) {
        LOG(INFO) << "have " << tracksMCTruth.getIndexedSize() << " track label(s)";
        // have to change the writer process as well but want to convert to the RootTreeWriter tool
        // at this occasion so we skip sending the labels for the moment
        //pc.outputs().snapshot(OutputRef{ "mclblout" }, tracksMCTruth);
      }

      validInputs.reset();
      if (processMC) {
        validMcInputs.reset();
        for (auto& mcInput : mcInputs) {
          mcInput.clear();
        }
      }
    };

    return processingFct;
  };

  auto createInputSpecs = [fanIn](bool makeMcInput) {
    std::vector<InputSpec> inputSpecs;
    for (size_t n = 0; n < fanIn; ++n) {
      std::string label = "input" + std::to_string(n);
      inputSpecs.emplace_back(InputSpec{ label, gDataOriginTPC, "CLUSTERNATIVE", n, Lifetime::Timeframe });

      if (makeMcInput) {
        label = "mclblin" + std::to_string(n);
        constexpr o2::header::DataDescription datadesc("CLNATIVEMCLBL");
        inputSpecs.emplace_back(InputSpec{ label, gDataOriginTPC, datadesc, n, Lifetime::Timeframe });
      }
    }
    return std::move(inputSpecs);
  };

  auto createOutputSpecs = [](bool makeMcOutput) {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{ { "output" }, gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe },
    };
    if (makeMcOutput) {
      OutputLabel label{ "mclblout" };
      constexpr o2::header::DataDescription datadesc("TRACKMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{ "tpc-tracker", // process id
                            { createInputSpecs(processMC) },
                            { createOutputSpecs(false /*create onece writer process has been changed*/) },
                            AlgorithmSpec(initFunction),
                            Options{
                              { "tracker-options", VariantType::String, "", { "Option string passed to tracker" } },
                            } };
}

} // namespace TPC
} // namespace o2
