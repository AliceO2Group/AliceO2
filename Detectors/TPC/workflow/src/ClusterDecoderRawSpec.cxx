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
#include "Framework/DataRefUtils.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/HardwareClusterDecoder.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <FairMQLogger.h>
#include <memory> // for make_shared
#include <vector>

using namespace o2::framework;

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

      auto outputDesc = Output{ "TPC", "CLUSTERNATIVE", 0, Lifetime::Timeframe };
      auto* target = pc.outputs().newChunk(outputDesc, totalSize).data;

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

  // We can split the output on sector level, but the DPL does not scale if there are too many specs
  // at the moment we have to create out specs for all individual data packages
  // its planned to support ranges of subSpecifications in the DPL
  auto createOutputSpec = []() {
    o2::framework::Outputs outputs;
    /**
    for (uint8_t sector = 0; sector < o2::TPC::Constants::MAXSECTOR; sector++) {
      auto subSpec = ClusterGroupAttribute{sector, 0}.getSubSpecification();
      outputs.emplace_back(OutputSpec{ "TPC", "CLUSTERNATIVE", subSpec, Lifetime::Timeframe });
    }
    */
    outputs.emplace_back(OutputSpec{ "TPC", "CLUSTERNATIVE", 0, Lifetime::Timeframe });

    return std::move(outputs);
  };

  return DataProcessorSpec{ "decoder",
                            { InputSpec{ "rawin", "TPC", "CLUSTERHW", 0, Lifetime::Timeframe } },
                            { createOutputSpec() },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
