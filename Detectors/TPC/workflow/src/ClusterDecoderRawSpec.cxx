// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterDecoderRawSpec.cxx
/// @author Matthias Richter
/// @since  2018-03-26
/// @brief  Processor spec for decoder of TPC raw cluster data

#include "ClusterDecoderRawSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/DataRefUtils.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/HardwareClusterDecoder.h"
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
/// convert incoming TPC raw to TPC native clusters
DataProcessorSpec getClusterDecoderRawSpec()
{
  auto initFunction = [](InitContext& ic) {
    // there is nothing to init at the moment
    auto decoder = std::make_shared<HardwareClusterDecoder>();

    auto processingFct = [decoder](ProcessingContext& pc) {
      // this will return a span of TPC clusters
      const auto& ref = pc.inputs().get("rawin");
      auto size = o2::framework::DataRefUtils::getPayloadSize(ref);
      if (ref.payload == nullptr) {
        return;
      }
      auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("rawin"));
      o2::header::Stack headerStack;
      if (sectorHeader) {
        o2::header::Stack actual{ *sectorHeader };
        std::swap(headerStack, actual);
      }

      std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList = {
        { reinterpret_cast<const ClusterHardwareContainer*>(ref.payload), size / 8192 }
      };
      std::vector<ClusterNativeContainer> cont;
      decoder->decodeClusters(inputList, cont);

      // FIXME: avoid the copy by snapshot: allocate array for max number of clusters and
      // update after writing
      size_t totalSize = 0;
      for (const auto& coll : cont) {
        const auto& groupAttribute = static_cast<const ClusterGroupAttribute>(coll);
        LOG(DEBUG) << "cluster native collection sector " << (int)groupAttribute.sector << ", global padow "
                   << (int)groupAttribute.globalPadRow << ": " << coll.clusters.size();
        totalSize += coll.getFlatSize() + sizeof(ClusterGroupHeader) - sizeof(ClusterGroupAttribute);
      }

      auto* target = pc.outputs().newChunk(OutputRef{ "clout", 0, std::move(headerStack) }, totalSize).data;

      for (const auto& coll : cont) {
        ClusterGroupHeader groupHeader(coll, coll.clusters.size());
        memcpy(target, &groupHeader, sizeof(groupHeader));
        target += sizeof(groupHeader);
        memcpy(target, coll.data(), coll.getFlatSize() - sizeof(ClusterGroupAttribute));
        target += coll.getFlatSize() - sizeof(ClusterGroupAttribute);
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  return DataProcessorSpec{ "decoder",
                            { InputSpec{ "rawin", gDataOriginTPC, "CLUSTERHW", 0, Lifetime::Timeframe } },
                            { OutputSpec{ { "clout" }, gDataOriginTPC, "CLUSTERNATIVE", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
