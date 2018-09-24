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

/// create the processor spec for TPC raw cluster decoder converting TPC raw to native clusters
/// Input: raw pages of TPC raw clusters
/// Output: vector of containers with clusters in ClusterNative format, one container per
/// (sector,globalPadRow)-address, the output is flattend in one single binary buffer
///
/// MC labels are received as MCLabelContainers
DataProcessorSpec getClusterDecoderRawSpec(bool sendMC)
{
  auto initFunction = [sendMC](InitContext& ic) {
    // there is nothing to init at the moment
    auto verbosity = 0;
    auto decoder = std::make_shared<HardwareClusterDecoder>();

    auto processingFct = [verbosity, decoder, sendMC](ProcessingContext& pc) {
      // this will return a span of TPC clusters
      const auto& ref = pc.inputs().get("rawin");
      auto size = o2::framework::DataRefUtils::getPayloadSize(ref);
      if (ref.payload == nullptr) {
        return;
      }

      // init the stacks for forwarding the sector header
      // FIXME check if there is functionality in the DPL to forward the stack
      // FIXME make one function
      o2::header::Stack rawHeaderStack;
      o2::header::Stack mcHeaderStack;
      if (sendMC) {
        auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("mclblin"));
        if (sectorHeader) {
          o2::header::Stack actual{ *sectorHeader };
          std::swap(mcHeaderStack, actual);
        }
      }
      auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("rawin"));
      if (sectorHeader) {
        o2::header::Stack actual{ *sectorHeader };
        std::swap(rawHeaderStack, actual);
      }

      // input to the decoder is a vector of raw pages description ClusterHardwareContainer,
      // each specified as a pair of pointer to ClusterHardwareContainer and the number
      // of pages in that buffer
      std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList = {
        { reinterpret_cast<const ClusterHardwareContainer*>(ref.payload), size / 8192 }
      };
      // MC labels are received as one container of labels in the sequence matching clusters
      // in the raw pages
      std::vector<MCLabelContainer> mcinList;
      std::vector<MCLabelContainer>* mcinListPointer = nullptr;
      if (sendMC) {
        auto mcin = pc.inputs().get<MCLabelContainer*>("mclblin");
        // FIXME: the decoder takes vector of MCLabelContainers as input and the retreived
        // input can not be moved because the input is const, so we have to copy
        mcinList.emplace_back(*mcin.get());
        mcinListPointer = &mcinList;
      }
      // output of the decoder is sorted in (sector,globalPadRow) coordinates, individual
      // containers are created for clusters and MC labels per (sector,globalPadRow) address
      std::vector<ClusterNativeContainer> cont;
      std::vector<MCLabelContainer> mcoutList;
      decoder->decodeClusters(inputList, cont, mcinListPointer, &mcoutList);

      // The output of clusters involves a copy to flatten the list of buffers and extend the
      // ClusterGroupAttribute struct containing sector and globalPadRow by the size of the collection.
      // FIXME: provide allocator to decoder to do the allocation in place, create ClusterGroupHeader
      //        directly
      // The vector of MC label containers is simply serialized
      size_t totalSize = 0;
      for (const auto& coll : cont) {
        const auto& groupAttribute = static_cast<const ClusterGroupAttribute>(coll);
        totalSize += coll.getFlatSize() + sizeof(ClusterGroupHeader) - sizeof(ClusterGroupAttribute);
      }

      auto* target = pc.outputs().newChunk(OutputRef{ "clusterout", 0, std::move(rawHeaderStack) }, totalSize).data;

      for (const auto& coll : cont) {
        if (verbosity > 0) {
          LOG(INFO) << "decoder " << std::setw(2) << sectorHeader->sector                             //
                    << ": decoded " << std::setw(4) << coll.clusters.size() << " clusters on sector " //
                    << std::setw(2) << (int)coll.sector << "[" << (int)coll.globalPadRow << "]";      //
        }
        ClusterGroupHeader groupHeader(coll, coll.clusters.size());
        memcpy(target, &groupHeader, sizeof(groupHeader));
        target += sizeof(groupHeader);
        memcpy(target, coll.data(), coll.getFlatSize() - sizeof(ClusterGroupAttribute));
        target += coll.getFlatSize() - sizeof(ClusterGroupAttribute);
      }
      if (sendMC) {
        if (verbosity > 0) {
          LOG(INFO) << "sending " << mcoutList.size() << " MC label containers" << std::endl;
        }
        // serialize the complete list of MC label containers
        pc.outputs().snapshot(OutputRef{ "mclblout", 0, std::move(mcHeaderStack) }, mcoutList);
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  auto createInputSpecs = [](bool makeMcInput) {
    std::vector<InputSpec> inputSpecs{
      InputSpec{ { "rawin" }, gDataOriginTPC, "CLUSTERHW", 0, Lifetime::Timeframe },
    };
    if (makeMcInput) {
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERMCLBL");
      inputSpecs.emplace_back(InputSpec{ "mclblin", gDataOriginTPC, datadesc, 0, Lifetime::Timeframe });
    }
    return std::move(inputSpecs);
  };

  auto createOutputSpecs = [](bool makeMcOutput) {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{ { "clusterout" }, gDataOriginTPC, "CLUSTERNATIVE", 0, Lifetime::Timeframe },
    };
    if (makeMcOutput) {
      OutputLabel label{ "mclblout" };
      // have to use a new data description, routing is only based on origin and decsription
      constexpr o2::header::DataDescription datadesc("CLNATIVEMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{ "decoder",
                            { createInputSpecs(sendMC) },
                            { createOutputSpecs(sendMC) },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
