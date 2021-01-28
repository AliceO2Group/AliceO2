// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   PublisherSpec.cxx
/// @author Matthias Richter
/// @since  2018-12-06
/// @brief  Processor spec for a reader of TPC data from ROOT file

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "TPCWorkflow/PublisherSpec.h"
#include "Headers/DataHeader.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include <memory> // for make_shared, make_unique, unique_ptr
#include <array>
#include <vector>
#include <utility>   // std::move
#include <stdexcept> //std::invalid_argument
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace tpc
{

/// create a processor spec
/// read data from multiple tree branches from ROOT file and publish
/// data are expected to be stored in separated branches per sector, the default
/// branch name is configurable, sector number is apended as extension '_n'
DataProcessorSpec createPublisherSpec(PublisherConf const& config, bool propagateMC, workflow_reader::Creator creator)
{
  if (config.tpcSectors.size() == 0 || config.outputIds.size() == 0) {
    throw std::invalid_argument("need TPC sector and output id configuration");
  }
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  enum struct SectorMode {
    Sector, // stored in sector branches
    Full,   // full TPC stored in one branch
  };
  struct ProcessAttributes {
    std::vector<int> sectors;
    std::vector<int> outputIds;
    std::vector<o2::header::DataHeader::SubSpecificationType> zeroLengthOutputs;
    uint64_t activeSectors = 0;
    std::array<std::shared_ptr<RootTreeReader>, NSectors> readers;
    bool terminateOnEod = false;
    bool finished = false;
    SectorMode sectorMode = SectorMode::Sector;
  };

  auto initFunction = [config, propagateMC, creator](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("infile");
    auto treename = ic.options().get<std::string>("treename");
    auto clbrName = ic.options().get<std::string>(config.databranch.option.c_str());
    auto mcbrName = ic.options().get<std::string>(config.mcbranch.option.c_str());
    auto nofEvents = ic.options().get<int>("nevents");
    auto publishingMode = nofEvents == -1 ? RootTreeReader::PublishingMode::Single : RootTreeReader::PublishingMode::Loop;

    // do a runtime check if the branch name without sector number suffix is found in the file
    // if found the publisher will publish the single data set at one output route and empty
    // messages at all the others
    auto checkSectorMode = [&filename, &treename, &clbrName]() -> SectorMode {
      std::unique_ptr<TFile> file(TFile::Open(filename.c_str()));
      if (file) {
        TTree* tree = reinterpret_cast<TTree*>(file->GetObjectChecked(treename.c_str(), "TTree"));
        if (tree) {
          const auto brlist = tree->GetListOfBranches();
          for (TObject const* entry : *brlist) {
            if (clbrName == entry->GetName()) {
              return SectorMode::Full;
            }
          }
        }
        file->Close();
      }
      return SectorMode::Sector;
    };

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      processAttributes->terminateOnEod = ic.options().get<bool>("terminate-on-eod");
      processAttributes->sectorMode = checkSectorMode();
      auto& sectors = processAttributes->sectors;
      auto& activeSectors = processAttributes->activeSectors;
      auto& readers = processAttributes->readers;
      auto& outputIds = processAttributes->outputIds;
      auto& sectorMode = processAttributes->sectorMode;

      sectors = config.tpcSectors;
      outputIds = config.outputIds;
      for (auto const& s : sectors) {
        // set the mask of active sectors
        if (s >= NSectors) {
          std::string message = std::string("invalid sector range specified, allowed 0-") + std::to_string(NSectors - 1);
          // FIXME should probably be FATAL, but this doesn't seem to be handled in the DPL control flow
          // at least the process is not marked dead in the DebugGUI
          LOG(ERROR) << message;
          throw std::invalid_argument(message);
        }
        activeSectors |= (uint64_t)0x1 << s;
      }

      // set up the tree interface
      // TODO: parallelism on sectors needs to be implemented as selector in the reader
      // the data is now in parallel branches, as first attempt use an array of readers
      auto outputId = outputIds.begin();
      for (auto const& sector : sectors) {
        o2::header::DataHeader::SubSpecificationType subSpec = *outputId;
        std::string sectorfile = filename;
        if (filename.find('%') != std::string::npos) {
          vector<char> formattedname(filename.length() + 10, 0);
          snprintf(formattedname.data(), formattedname.size() - 1, filename.c_str(), sector);
          sectorfile = formattedname.data();
        }
        std::string clusterbranchname = clbrName;
        std::string mcbranchname = mcbrName;
        if (sectorMode == SectorMode::Sector) {
          clusterbranchname += "_" + std::to_string(sector);
          mcbranchname += "_" + std::to_string(sector);
        }
        readers[sector] = creator(treename.c_str(),   // tree name
                                  sectorfile.c_str(), // input file name
                                  nofEvents,          // number of entries to publish
                                  publishingMode,
                                  subSpec,
                                  clusterbranchname.c_str(), // name of data branch
                                  mcbranchname.c_str(),      // name of mc label branch
                                  config.hook);
        if (sectorMode == SectorMode::Full) {
          break;
        }
        if (++outputId == outputIds.end()) {
          outputId = outputIds.begin();
        }
      }
      if (sectorMode == SectorMode::Full) {
        // the slot of the first configured sector is used to publish the full set, all others removed
        sectors.resize(1);
        // the data will be published at first configured output id, zero-length data on all other output ids
        processAttributes->zeroLengthOutputs.assign(++outputId, outputIds.end());
      }
    }

    // set up the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    // FIXME: wanted to use it = sectors.begin() in the variable capture but the iterator
    // is const and can not be incremented
    auto processingFct = [processAttributes, config](ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      bool eos = false;
      auto const& sectors = processAttributes->sectors;
      for (auto const& sector : sectors) {
        auto& activeSectors = processAttributes->activeSectors;
        auto& readers = processAttributes->readers;
        o2::tpc::TPCSectorHeader header{sector};
        if (processAttributes->sectorMode == SectorMode::Full) {
          header.sectorBits = activeSectors;
        }
        header.activeSectors = activeSectors;
        auto& r = *(readers[sector].get());

        // increment the reader and invoke it for the processing context
        if (r.next()) {
          // there is data, run the reader
          r(pc, header);
        } else {
          // no more data, delete the reader
          readers[sector].reset();
          eos = true;
        }
      }

      if (eos) {
        processAttributes->finished = true;
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      } else {
        // publish empty events
        auto dto = DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
        auto mco = DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);
        o2::tpc::TPCSectorHeader header{0};
        header.sectorBits = 0;
        header.activeSectors = processAttributes->activeSectors;
        for (auto const& subSpec : processAttributes->zeroLengthOutputs) {
          pc.outputs().make<char>({dto.origin, dto.description, subSpec, Lifetime::Timeframe, {header}});
          if (pc.outputs().isAllowed({mco.origin, mco.description, subSpec})) {
            pc.outputs().make<char>({mco.origin, mco.description, subSpec, Lifetime::Timeframe, {header}});
          }
        }
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  auto createOutputSpecs = [&config, propagateMC]() {
    std::vector<OutputSpec> outputSpecs;
    for (size_t n = 0; n < config.outputIds.size(); ++n) {
      o2::header::DataHeader::SubSpecificationType subSpec = config.outputIds[n];
      auto dto = DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
      auto mco = DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);
      outputSpecs.emplace_back(OutputSpec{{"output"}, dto.origin, dto.description, subSpec, Lifetime::Timeframe});
      if (propagateMC) {
        outputSpecs.emplace_back(OutputSpec{{"outputMC"}, mco.origin, mco.description, subSpec, Lifetime::Timeframe});
      }
    }
    return std::move(outputSpecs);
  };

  auto& dtb = config.databranch;
  auto& mcb = config.mcbranch;
  return DataProcessorSpec{config.processName.c_str(),
                           Inputs{}, // no inputs
                           {createOutputSpecs()},
                           AlgorithmSpec(initFunction),
                           Options{
                             {"infile", VariantType::String, config.defaultFileName.c_str(), {"Name of the input file"}},
                             {"treename", VariantType::String, config.defaultTreeName.c_str(), {"Name of input tree"}},
                             {dtb.option.c_str(), VariantType::String, dtb.defval.c_str(), {dtb.help.c_str()}},
                             {mcb.option.c_str(), VariantType::String, mcb.defval.c_str(), {mcb.help.c_str()}},
                             {"nevents", VariantType::Int, -1, {"number of events to run"}},
                             {"terminate-on-eod", VariantType::Bool, true, {"terminate on end-of-data"}},
                           }};
}
} // end namespace tpc
} // end namespace o2
