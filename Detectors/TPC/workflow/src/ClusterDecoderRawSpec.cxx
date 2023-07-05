// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include "Framework/InputRecordWalker.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/HardwareClusterDecoder.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <fairlogger/Logger.h>
#include <set>
#include <memory> // for make_shared
#include <vector>
#include <map>
#include <cassert>
#include <iomanip>
#include <string>
#include <numeric>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::dataformats;

namespace o2
{
namespace tpc
{
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
    int verbosity = 0;
    bool sendMC = false;
  };

  auto initFunction = [sendMC](InitContext& ic) {
    // there is nothing to init at the moment
    auto processAttributes = std::make_shared<ProcessAttributes>();
    processAttributes->decoder = std::make_unique<HardwareClusterDecoder>();
    processAttributes->sendMC = sendMC;

    auto processSectorFunction = [processAttributes](ProcessingContext& pc, DataRef const& ref, DataRef const& mclabelref) {
      auto& decoder = processAttributes->decoder;
      auto& verbosity = processAttributes->verbosity;
      auto& activeInputs = processAttributes->activeInputs;
      // this will return a span of TPC clusters
      auto size = o2::framework::DataRefUtils::getPayloadSize(ref);
      auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      o2::header::DataHeader::SubSpecificationType fanSpec = dataHeader->subSpecification;

      // init the stacks for forwarding the sector header
      // FIXME check if there is functionality in the DPL to forward the stack
      // FIXME make one function
      o2::header::Stack rawHeaderStack;
      o2::header::Stack mcHeaderStack;
      o2::tpc::TPCSectorHeader const* sectorHeaderMC = nullptr;
      if (DataRefUtils::isValid(mclabelref)) {
        sectorHeaderMC = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(mclabelref);
        if (sectorHeaderMC) {
          o2::header::Stack actual{*sectorHeaderMC};
          std::swap(mcHeaderStack, actual);
          if (sectorHeaderMC->sector() < 0) {
            pc.outputs().snapshot(Output{gDataOriginTPC, DataDescription("CLNATIVEMCLBL"), fanSpec, Lifetime::Timeframe, std::move(mcHeaderStack)}, fanSpec);
          }
        }
      }
      auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
      if (sectorHeader) {
        o2::header::Stack actual{*sectorHeader};
        std::swap(rawHeaderStack, actual);
        if (sectorHeader->sector() < 0) {
          pc.outputs().snapshot(Output{gDataOriginTPC, DataDescription("CLUSTERNATIVE"), fanSpec, Lifetime::Timeframe, std::move(rawHeaderStack)}, fanSpec);
          return;
        }
      }
      assert(sectorHeaderMC == nullptr || sectorHeader->sector() == sectorHeaderMC->sector());

      // input to the decoder is a vector of raw pages description ClusterHardwareContainer,
      // each specified as a pair of pointer to ClusterHardwareContainer and the number
      // of pages in that buffer
      // FIXME: better description of the raw page
      size_t nPages = size / 8192;
      std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList;
      if (verbosity > 0 && !DataRefUtils::isValid(mclabelref)) {
        LOG(info) << "Decoder input: " << size << ", " << nPages << " pages for sector " << sectorHeader->sector();
      }

      // MC labels are received as one container of labels in the sequence matching clusters
      // in the raw pages
      std::vector<ConstMCLabelContainer> mcinCopiesFlat;
      std::vector<ConstMCLabelContainerView> mcinCopiesFlatView;
      ConstMCLabelContainerView mcin;
      if (DataRefUtils::isValid(mclabelref)) {
        mcin = pc.inputs().get<gsl::span<char>>(mclabelref);
        mcinCopiesFlat.resize(nPages);
        mcinCopiesFlatView.reserve(nPages);
        if (verbosity > 0) {
          LOG(info) << "Decoder input: " << size << ", " << nPages << " pages, " << mcin.getIndexedSize() << " MC label sets for sector " << sectorHeader->sector();
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
        MCLabelContainer mcinCopy;
        inputList.emplace_back(reinterpret_cast<const ClusterHardwareContainer*>(ref.payload + page * 8192), 1);
        const ClusterHardwareContainer& container = *(inputList.back().first);
        if (verbosity > 1) {
          LOG(info) << "Decoder input in page " << std::setw(2) << page << ": "     //
                    << "CRU " << std::setw(3) << container.CRU << " "               //
                    << std::setw(3) << container.numberOfClusters << " cluster(s)"; //
        }
        totalNumberOfClusters += container.numberOfClusters;
        if (mcin.getBuffer().size()) {
          for (size_t mccopyPos = 0;
               mccopyPos < container.numberOfClusters && mcinPos < mcin.getIndexedSize();
               mccopyPos++, mcinPos++) {
            for (auto const& label : mcin.getLabels(mcinPos)) {
              mcinCopy.addElement(mccopyPos, label);
            }
          }
        }
        mcinCopy.flatten_to(mcinCopiesFlat[page]);
        mcinCopiesFlatView.emplace_back(mcinCopiesFlat[page]);
      }
      // FIXME: introduce error handling policy: throw, ignore, warn
      //assert(!mcin || mcinPos == mcin->getIndexedSize());
      if (mcin.getBuffer().size() && mcinPos != totalNumberOfClusters) {
        LOG(error) << "inconsistent number of MC label objects processed"
                   << ", expecting MC label objects for " << totalNumberOfClusters << " cluster(s)"
                   << ", got " << mcin.getIndexedSize();
      }
      // output of the decoder is sorted in (sector,globalPadRow) coordinates, individual
      // containers are created for clusters and MC labels per (sector,globalPadRow) address
      char* outputBuffer = nullptr;
      auto outputAllocator = [&pc, &fanSpec, &outputBuffer, &rawHeaderStack](size_t size) -> char* {
        outputBuffer = pc.outputs().newChunk(Output{gDataOriginTPC, DataDescription("CLUSTERNATIVE"), fanSpec, Lifetime::Timeframe, std::move(rawHeaderStack)}, size).data();
        return outputBuffer;
      };
      MCLabelContainer mcout;
      decoder->decodeClusters(inputList, outputAllocator, (mcin.getBuffer().size() ? &mcinCopiesFlatView : nullptr), &mcout);

      // TODO: reestablish the logging messages on the raw buffer
      // if (verbosity > 1) {
      //   LOG(info) << "decoder " << std::setw(2) << sectorHeader->sector()                             //
      //             << ": decoded " << std::setw(4) << coll.clusters.size() << " clusters on sector " //
      //             << std::setw(2) << (int)coll.sector << "[" << (int)coll.globalPadRow << "]";      //
      // }

      if (DataRefUtils::isValid(mclabelref)) {
        if (verbosity > 0) {
          LOG(info) << "sending " << mcout.getIndexedSize()
                    << " label object(s)" << std::endl;
        }
        // serialize the complete list of MC label containers
        ConstMCLabelContainer labelsFlat;
        mcout.flatten_to(labelsFlat);
        pc.outputs().snapshot(Output{gDataOriginTPC, DataDescription("CLNATIVEMCLBL"), fanSpec, Lifetime::Timeframe, std::move(mcHeaderStack)}, labelsFlat);
      }
    };

    auto processingFct = [processAttributes, processSectorFunction](ProcessingContext& pc) {
      struct SectorInputDesc {
        DataRef dataref;
        DataRef mclabelref;
      };
      // loop over all inputs and their parts and associate data with corresponding mc truth data
      // by the subspecification
      std::map<int, SectorInputDesc> inputs;
      std::vector<InputSpec> filter = {
        {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERHW"}, Lifetime::Timeframe},
        {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERHWMCLBL"}, Lifetime::Timeframe},
      };
      for (auto const& inputRef : InputRecordWalker(pc.inputs(), filter)) {
        auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(inputRef);
        if (sectorHeader == nullptr) {
          LOG(error) << "sector header missing on header stack for input on " << inputRef.spec->binding;
          continue;
        }
        const int sector = sectorHeader->sector();
        if (DataRefUtils::match(inputRef, {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERHW"}})) {
          inputs[sector].dataref = inputRef;
        }
        if (DataRefUtils::match(inputRef, {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERHWMCLBL"}})) {
          inputs[sector].mclabelref = inputRef;
        }
      }
      for (auto const& input : inputs) {
        if (processAttributes->sendMC && !DataRefUtils::isValid(input.second.mclabelref)) {
          throw std::runtime_error("missing the required MC label data for sector " + std::to_string(input.first));
        }
        processSectorFunction(pc, input.second.dataref, input.second.mclabelref);
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
