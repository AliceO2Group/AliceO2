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

#include "TPCWorkflow/ClusterDecoderRawSpec.h"
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
namespace tpc
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
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  using DataDescription = o2::header::DataDescription;
  std::string processorName = "tpc-cluster-decoder";
  struct ProcessAttributes {
    std::unique_ptr<HardwareClusterDecoder> decoder;
    std::set<o2::header::DataHeader::SubSpecificationType> activeInputs;
    bool readyToQuit = false;
    int verbosity = 0;
    bool sendMC = false;
  };

  auto initFunction = [sendMC](InitContext& ic) {
    // there is nothing to init at the moment
    auto processAttributes = std::make_shared<ProcessAttributes>();
    processAttributes->decoder = std::make_unique<HardwareClusterDecoder>();
    processAttributes->sendMC = sendMC;

    auto processSectorFunction = [processAttributes](ProcessingContext& pc, std::string inputKey, std::string labelKey) -> bool {
      auto& decoder = processAttributes->decoder;
      auto& verbosity = processAttributes->verbosity;
      auto& activeInputs = processAttributes->activeInputs;
      // this will return a span of TPC clusters
      const auto& ref = pc.inputs().get(inputKey.c_str());
      auto size = o2::framework::DataRefUtils::getPayloadSize(ref);
      auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().get(inputKey.c_str()));
      o2::header::DataHeader::SubSpecificationType fanSpec = dataHeader->subSpecification;

      // init the stacks for forwarding the sector header
      // FIXME check if there is functionality in the DPL to forward the stack
      // FIXME make one function
      o2::header::Stack rawHeaderStack;
      o2::header::Stack mcHeaderStack;
      o2::tpc::TPCSectorHeader const* sectorHeaderMC = nullptr;
      if (!labelKey.empty()) {
        sectorHeaderMC = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(pc.inputs().get(labelKey.c_str()));
        if (sectorHeaderMC) {
          o2::header::Stack actual{*sectorHeaderMC};
          std::swap(mcHeaderStack, actual);
          if (sectorHeaderMC->sector < 0) {
            pc.outputs().snapshot(Output{gDataOriginTPC, DataDescription("CLNATIVEMCLBL"), fanSpec, Lifetime::Timeframe, std::move(mcHeaderStack)}, fanSpec);
          }
        }
      }
      auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(pc.inputs().get(inputKey.c_str()));
      if (sectorHeader) {
        o2::header::Stack actual{*sectorHeader};
        std::swap(rawHeaderStack, actual);
        if (sectorHeader->sector < 0) {
          pc.outputs().snapshot(Output{gDataOriginTPC, DataDescription("CLUSTERNATIVE"), fanSpec, Lifetime::Timeframe, std::move(rawHeaderStack)}, fanSpec);
          return (sectorHeader->sector == -1);
        }
      }
      assert(sectorHeaderMC == nullptr || sectorHeader->sector == sectorHeaderMC->sector);

      // input to the decoder is a vector of raw pages description ClusterHardwareContainer,
      // each specified as a pair of pointer to ClusterHardwareContainer and the number
      // of pages in that buffer
      // FIXME: better description of the raw page
      size_t nPages = size / 8192;
      std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList;
      if (verbosity > 0 && labelKey.empty()) {
        LOG(INFO) << "Decoder input: " << size << ", " << nPages << " pages for sector " << sectorHeader->sector;
      }

      // MC labels are received as one container of labels in the sequence matching clusters
      // in the raw pages
      std::vector<MCLabelContainer> mcinCopies;
      std::unique_ptr<const MCLabelContainer> mcin;
      if (!labelKey.empty()) {
        mcin = std::move(pc.inputs().get<MCLabelContainer*>(labelKey.c_str()));
        mcinCopies.resize(nPages);
        if (verbosity > 0) {
          LOG(INFO) << "Decoder input: " << size << ", " << nPages << " pages, " << mcin->getIndexedSize() << " MC label sets for sector " << sectorHeader->sector;
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
        if (verbosity > 1) {
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
      char* outputBuffer = nullptr;
      auto outputAllocator = [&pc, &fanSpec, &outputBuffer, &rawHeaderStack](size_t size) -> char* {
        outputBuffer = pc.outputs().newChunk(Output{gDataOriginTPC, DataDescription("CLUSTERNATIVE"), fanSpec, Lifetime::Timeframe, std::move(rawHeaderStack)}, size).data();
        return outputBuffer;
      };
      std::vector<MCLabelContainer> mcoutList;
      decoder->decodeClusters(inputList, outputAllocator, (mcin ? &mcinCopies : nullptr), &mcoutList);

      // TODO: reestablish the logging messages on the raw buffer
      // if (verbosity > 1) {
      //   LOG(INFO) << "decoder " << std::setw(2) << sectorHeader->sector                             //
      //             << ": decoded " << std::setw(4) << coll.clusters.size() << " clusters on sector " //
      //             << std::setw(2) << (int)coll.sector << "[" << (int)coll.globalPadRow << "]";      //
      // }

      if (!labelKey.empty()) {
        if (verbosity > 0) {
          LOG(INFO) << "sending " << mcoutList.size() << " MC label container(s) with in total "
                    << std::accumulate(mcoutList.begin(), mcoutList.end(), size_t(0), [](size_t l, auto const& r) { return l + r.getIndexedSize(); })
                    << " label object(s)" << std::endl;
        }
        // serialize the complete list of MC label containers
        pc.outputs().snapshot(Output{gDataOriginTPC, DataDescription("CLNATIVEMCLBL"), fanSpec, Lifetime::Timeframe, std::move(mcHeaderStack)}, mcoutList);
      }
      return false;
    };

    auto processingFct = [processAttributes, processSectorFunction](ProcessingContext& pc) {
      if (processAttributes->readyToQuit) {
        return;
      }

      struct SectorInputDesc {
        std::string inputKey = "";
        std::string labelKey = "";
      };
      std::map<o2::header::DataHeader::SubSpecificationType, SectorInputDesc> inputs;
      for (auto const& inputRef : pc.inputs()) {
        if (pc.inputs().isValid(inputRef.spec->binding) == false) {
          // this input slot is empty
          continue;
        }
        // loop over all inputs and associate data and mc channels by the subspecification. DPL makes sure that
        // each origin/description/subspecification identifier is only once in the inputs, no other overwrite
        // protection in the list of keys.
        auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
        assert(dataHeader);
        if (dataHeader->dataOrigin == gDataOriginTPC && dataHeader->dataDescription == DataDescription("CLUSTERHW")) {
          inputs[dataHeader->subSpecification].inputKey = inputRef.spec->binding;
        } else if (dataHeader->dataOrigin == gDataOriginTPC && dataHeader->dataDescription == DataDescription("CLUSTERHWMCLBL")) {
          inputs[dataHeader->subSpecification].labelKey = inputRef.spec->binding;
        }
      }
      if (processAttributes->sendMC) {
        // need to check whether data-MC pairs are complete
        for (auto const& input : inputs) {
          if (input.second.inputKey.empty() || input.second.labelKey.empty()) {
            // we wait for the data set to be complete next time
            return;
          }
        }
      }
      // will stay true if all inputs signal finished
      // this implies that all inputs are always processed together, when changing this policy the
      // status of each input needs to be kept individually
      bool finished = true;
      for (auto const& input : inputs) {
        finished = finished & processSectorFunction(pc, input.second.inputKey, input.second.labelKey);
      }
      if (finished) {
        // got EOD on all inputs
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
        processAttributes->readyToQuit = true;
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  auto createInputSpecs = [](bool makeMcInput) {
    std::vector<InputSpec> inputSpecs{
      InputSpec{{"rawin"}, gDataOriginTPC, "CLUSTERHW", 0, Lifetime::Timeframe},
    };
    if (makeMcInput) {
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERHWMCLBL");
      inputSpecs.emplace_back(InputSpec{"mclblin", gDataOriginTPC, datadesc, 0, Lifetime::Timeframe});
    }
    return std::move(inputSpecs);
  };

  auto createOutputSpecs = [](bool makeMcOutput) {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{{"clusterout"}, gDataOriginTPC, "CLUSTERNATIVE", 0, Lifetime::Timeframe},
    };
    if (makeMcOutput) {
      OutputLabel label{"mclblout"};
      // have to use a new data description, routing is only based on origin and decsription
      constexpr o2::header::DataDescription datadesc("CLNATIVEMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{processorName,
                           {createInputSpecs(sendMC)},
                           {createOutputSpecs(sendMC)},
                           AlgorithmSpec(initFunction)};
}

} // namespace tpc
} // namespace o2
