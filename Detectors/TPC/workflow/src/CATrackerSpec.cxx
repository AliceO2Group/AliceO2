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
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/TPCCATracking.h"
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
  auto initFunction = [](InitContext& ic) {
    auto options = ic.options().get<std::string>("tracker-options");

    using ClusterGroupParser = o2::algorithm::ForwardParser<o2::TPC::ClusterGroupHeader>;
    auto parser = std::make_shared<ClusterGroupParser>();
    auto tracker = std::make_shared<o2::TPC::TPCCATracking>();
    tracker->initialize(options.c_str());

    auto processingFct = [parser, tracker](ProcessingContext& pc) {
      ClusterNativeAccessFullTPC clusterIndex;
      memset(&clusterIndex, 0, sizeof(clusterIndex));

      for (const auto& ref : pc.inputs()) {
        auto size = o2::framework::DataRefUtils::getPayloadSize(ref);
        LOG(INFO) << "  " << *(ref.spec) << ", size " << size;
        parser->parse(ref.payload, size,
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
