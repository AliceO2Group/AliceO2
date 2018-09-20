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

DataProcessorSpec getCATrackerSpec()
{
  constexpr static size_t NSectors = o2::TPC::Sector::MAXSECTOR;
  using ClusterGroupParser = o2::algorithm::ForwardParser<o2::TPC::ClusterGroupHeader>;
  struct ProcessAttributes {
    // the input comes in individual calls and we need to buffer until
    // data set is complete, have to think about a DPL feature to take
    // ownership of an input
    std::array<std::vector<unsigned char>, NSectors> inputs;
    std::bitset<NSectors> validInputs = 0;
    std::unique_ptr<ClusterGroupParser> parser;
    std::unique_ptr<o2::TPC::TPCCATracking> tracker;
    int verbosity = 1;
  };

  auto initFunction = [](InitContext& ic) {
    auto options = ic.options().get<std::string>("tracker-options");

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      parser = std::make_unique<ClusterGroupParser>();
      tracker = std::make_unique<o2::TPC::TPCCATracking>();
      tracker->initialize(options.c_str());
      processAttributes->validInputs.reset();
    }

    auto processingFct = [processAttributes](ProcessingContext& pc) {
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      auto& validInputs = processAttributes->validInputs;
      auto& inputs = processAttributes->inputs;
      uint64_t activeSectors = 0;
      auto& verbosity = processAttributes->verbosity;

      // we can later extend this to multiple inputs
      std::vector<std::string> inputLabels = { "input" };
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

      if (activeSectors == 0 || (activeSectors & validInputs.to_ulong()) != activeSectors) {
        // not all sectors available
        // FIXME: do we need to send something
        return;
      }
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

                        return true;
                      });
      }

      std::vector<TrackTPC> tracks;
      int retVal = tracker->runTracking(clusterIndex, &tracks, nullptr);
      if (retVal != 0) {
        // FIXME: error policy
        LOG(ERROR) << "tracker returned error code " << retVal;
      }
      LOG(INFO) << "found " << tracks.size() << " track(s)";
      pc.outputs().snapshot(OutputRef{ "output" }, tracks);
    };

    return processingFct;
  };

  return DataProcessorSpec{ "tracker", // process id
                            { InputSpec{ "input", "TPC", "CLUSTERNATIVE", 0, Lifetime::Timeframe } },
                            { OutputSpec{ { "output" }, gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
