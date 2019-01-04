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
#include "Framework/ControlService.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/HardwareClusterDecoder.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <FairMQLogger.h>
#include <memory> // for make_shared
#include <vector>
#include <cassert>
#include <iomanip>
#include <string>
#include <numeric>

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
  std::string processorName = "tpc-cluster-decoder";

  auto initFunction = [sendMC](InitContext& ic) {
    // there is nothing to init at the moment
    auto verbosity = 0;
    auto decoder = std::make_shared<HardwareClusterDecoder>();

    auto processingFct = [verbosity, decoder, sendMC](ProcessingContext& pc) {
      static bool finished = false;
      if (finished) {
        return;
      }
      // this will return a span of TPC clusters
      const auto& ref = pc.inputs().get("rawin");
      auto size = o2::framework::DataRefUtils::getPayloadSize(ref);
      if (ref.payload == nullptr) {
        return;
      }
      auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().get("rawin"));
      o2::header::DataHeader::SubSpecificationType fanSpec = dataHeader->subSpecification;

      // init the stacks for forwarding the sector header
      // FIXME check if there is functionality in the DPL to forward the stack
      // FIXME make one function
      o2::header::Stack rawHeaderStack;
      o2::header::Stack mcHeaderStack;
      o2::TPC::TPCSectorHeader const* sectorHeaderMC = nullptr;
      if (sendMC) {
        sectorHeaderMC = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("mclblin"));
        if (sectorHeaderMC) {
          o2::header::Stack actual{ *sectorHeaderMC };
          std::swap(mcHeaderStack, actual);
          if (sectorHeaderMC->sector < 0) {
            pc.outputs().snapshot(OutputRef{ "mclblout", fanSpec, std::move(mcHeaderStack) }, fanSpec);
          }
        }
      }
      auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("rawin"));
      if (sectorHeader) {
        o2::header::Stack actual{ *sectorHeader };
        std::swap(rawHeaderStack, actual);
        if (sectorHeader->sector < 0) {
          pc.outputs().snapshot(OutputRef{ "clusterout", fanSpec, std::move(rawHeaderStack) }, fanSpec);
          if (sectorHeader->sector == -1) {
            // got EOD
            finished = true;
            pc.services().get<ControlService>().readyToQuit(false);
          }
          return;
        }
      }
      assert(sectorHeaderMC == nullptr || sectorHeader->sector == sectorHeaderMC->sector);

      // input to the decoder is a vector of raw pages description ClusterHardwareContainer,
      // each specified as a pair of pointer to ClusterHardwareContainer and the number
      // of pages in that buffer
      // FIXME: better description of the raw page
      size_t nPages = size / 8192;
      std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList;

      // MC labels are received as one container of labels in the sequence matching clusters
      // in the raw pages
      std::vector<MCLabelContainer> mcinCopies;
      std::unique_ptr<const MCLabelContainer> mcin;
      if (sendMC) {
        mcin = std::move(pc.inputs().get<MCLabelContainer*>("mclblin"));
        mcinCopies.resize(nPages);
        if (verbosity > 0) {
          LOG(INFO) << "Decoder input: " << size << ", " << nPages << " pages, " << mcin->getIndexedSize() << " MC label sets";
        }
      }

      // FIXME: the decoder takes vector of MCLabelContainers as input and the retreived
      // input can not be moved because the input is const, so we have to copy
      // Furthermore, the current decoder implementation supports handling of MC labels
      // only for single 8kb pages. So we have to add the raw pages individually and create
      // MC label containers for the corresponding clusters.
      size_t mcinPos = 0;
      size_t totalNumberOfClusters = 0;
      for (size_t page = 0; page < nPages; page++) {
        inputList.emplace_back(reinterpret_cast<const ClusterHardwareContainer*>(ref.payload + page * 8192), 1);
        const ClusterHardwareContainer& container = *(inputList.back().first);
        if (verbosity > 0) {
          LOG(INFO) << "Decoder input in page " << std::setw(2) << page << ": "     //
                    << "CRU " << std::setw(3) << container.CRU << " "               //
                    << std::setw(3) << container.numberOfClusters << " cluster(s)"; //
        }
        totalNumberOfClusters += container.numberOfClusters;
        if (mcin) {
          for (size_t mccopyPos = 0;
               mccopyPos < container.numberOfClusters && mcinPos < mcin->getIndexedSize();
               mccopyPos++, mcinPos++) {
            for (auto const& label : mcin->getLabels(mcinPos)) {
              mcinCopies[page].addElement(mccopyPos, label);
            }
          }
        }
      }
      // FIXME: introduce error handling policy: throw, ignore, warn
      //assert(!mcin || mcinPos == mcin->getIndexedSize());
      if (mcin && mcinPos != totalNumberOfClusters) {
        LOG(ERROR) << "inconsistent number of MC label objects processed"
                   << ", expecting MC label objects for " << totalNumberOfClusters << " cluster(s)"
                   << ", got " << mcin->getIndexedSize();
      }
      // output of the decoder is sorted in (sector,globalPadRow) coordinates, individual
      // containers are created for clusters and MC labels per (sector,globalPadRow) address
      std::vector<ClusterNativeContainer> cont;
      std::vector<MCLabelContainer> mcoutList;
      decoder->decodeClusters(inputList, cont, (mcin ? &mcinCopies : nullptr), &mcoutList);

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

      auto* target = pc.outputs().newChunk(OutputRef{ "clusterout", fanSpec, std::move(rawHeaderStack) }, totalSize).data;

      for (const auto& coll : cont) {
        if (verbosity > 1) {
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
          LOG(INFO) << "sending " << mcoutList.size() << " MC label container(s) with in total "
                    << std::accumulate(mcoutList.begin(), mcoutList.end(), size_t(0), [](size_t l, auto const& r) { return l + r.getIndexedSize(); })
                    << " label object(s)" << std::endl;
        }
        // serialize the complete list of MC label containers
        pc.outputs().snapshot(OutputRef{ "mclblout", fanSpec, std::move(mcHeaderStack) }, mcoutList);
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
      constexpr o2::header::DataDescription datadesc("CLUSTERHWMCLBL");
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

  return DataProcessorSpec{ processorName,
                            { createInputSpecs(sendMC) },
                            { createOutputSpecs(sendMC) },
                            AlgorithmSpec(initFunction) };
}

} // namespace TPC
} // namespace o2
