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

#include "Framework/ControlService.h"
#include "PublisherSpec.h"
#include "Headers/DataHeader.h"
#include "Utils/RootTreeReader.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include <memory> // for make_shared, make_unique, unique_ptr
#include <array>
#include <vector>
#include <utility>   // std::move
#include <stdexcept> //std::invalid_argument

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace TPC
{

/// create a processor spec
/// read data from multiple tree branches from ROOT file and publish
/// data are expected to be stored in separated branches per sector, the default
/// branch name is configurable, sector number is apended as extension '_n'
DataProcessorSpec getPublisherSpec(PublisherConf const& config, bool propagateMC)
{
  if (config.tpcSectors.size() == 0 || config.fanOut == 0) {
    throw std::invalid_argument("need TPC sector configuration and fan out size");
  }
  constexpr static size_t NSectors = o2::TPC::Sector::MAXSECTOR;
  struct ProcessAttributes {
    size_t nParallelReaders = 1;
    std::vector<int> sectors;
    uint64_t activeSectors = 0;
    std::array<std::shared_ptr<RootTreeReader>, NSectors> readers;
    bool terminateOnEod = false;
    bool finished = false;
  };

  auto initFunction = [config, propagateMC](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("infile");
    auto treename = ic.options().get<std::string>("treename");
    auto clbrName = ic.options().get<std::string>(config.databranch.option.c_str());
    auto mcbrName = ic.options().get<std::string>(config.mcbranch.option.c_str());
    auto nofEvents = ic.options().get<int>("nevents");

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      processAttributes->nParallelReaders = config.fanOut;
      processAttributes->terminateOnEod = ic.options().get<bool>("terminate-on-eod");
      auto& sectors = processAttributes->sectors;
      auto& activeSectors = processAttributes->activeSectors;
      auto& readers = processAttributes->readers;

      sectors = config.tpcSectors;
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
      constexpr auto persistency = Lifetime::Timeframe;
      o2::header::DataHeader::SubSpecificationType lane = 0;
      for (size_t sector = 0; sector < NSectors; ++sector) {
        if ((activeSectors & ((uint64_t)0x1 << sector)) == 0) {
          continue;
        }
        std::string clusterbranchname = clbrName + "_" + std::to_string(sector);
        std::string mcbranchname = mcbrName + "_" + std::to_string(sector);
        auto& dto = config.dataoutput;
        auto& mco = config.mcoutput;
        if (propagateMC) {
          readers[sector] = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                             nofEvents,        // number of entries to publish
                                                             filename.c_str(), // input file name
                                                             Output{ dto.origin, dto.description, lane, persistency },
                                                             clusterbranchname.c_str(), // name of cluster branch
                                                             Output{ mco.origin, mco.description, lane, persistency },
                                                             mcbranchname.c_str() // name of mc label branch
                                                             );
        } else {
          readers[sector] = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                             nofEvents,        // number of entries to publish
                                                             filename.c_str(), // input file name
                                                             Output{ dto.origin, dto.description, lane, persistency },
                                                             clusterbranchname.c_str() // name of cluster branch
                                                             );
        }
        if (++lane >= processAttributes->nParallelReaders) {
          lane = 0;
        }
      }
    }

    // set up the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    // FIXME: wanted to use it = sectors.begin() in the variable capture but the iterator
    // is const and can not be incremented
    auto processingFct = [processAttributes, index = std::make_shared<int>(0), propagateMC](ProcessingContext& pc)
    {
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &index, &pc, propagateMC]() {
        auto& sectors = processAttributes->sectors;
        auto& activeSectors = processAttributes->activeSectors;
        auto& readers = processAttributes->readers;
        while (*index < sectors.size() && !readers[sectors[*index]]) {
          // probably more efficient to use a vector of valid readers instead of the fixed array with
          // possibly invalid entries
          ++(*index);
        }
        if (*index == sectors.size()) {
          // there is no valid reader for this call
          return false;
        }
        auto sector = sectors[*index];
        o2::TPC::TPCSectorHeader header{ sector };
        header.activeSectors = activeSectors;
        auto& r = *(readers[sector].get());

        // increment the reader and invoke it for the processing context
        if (r.next()) {
          // there is data, run the reader
          r(pc, header);
        } else {
          // no more data, delete the reader
          readers[sector].reset();
          return false;
        }
        ++(*index);
        return true;
      };

      if (*index >= processAttributes->sectors.size()) {
        *index = 0;
      }
      int operation = -2;
      for (size_t lane = 0; lane < processAttributes->nParallelReaders; ++lane) {
        if (!publish()) {
          // need to publish a dummy packet
          // FIXME define and use flags in the TPCSectorHeader, for now using the same schema as
          // in digitizer workflow, -1 -> end of data, -2 noop
          if (lane == 0) {
            operation = -1;
          }
          o2::TPC::TPCSectorHeader header{ operation };
          pc.outputs().snapshot(OutputRef{ "output", lane, { header } }, lane);
          if (propagateMC) {
            pc.outputs().snapshot(OutputRef{ "outputMC", lane, { header } }, lane);
          }
        }
      }

      if ((processAttributes->finished = (operation == -1)) && processAttributes->terminateOnEod) {
        pc.services().get<ControlService>().readyToQuit(false);
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  auto createOutputSpecs = [&config, propagateMC]() {
    std::vector<OutputSpec> outputSpecs;
    for (size_t n = 0; n < config.fanOut; ++n) {
      outputSpecs.emplace_back(OutputSpec{ { "output" }, config.dataoutput.origin, config.dataoutput.description, n, Lifetime::Timeframe });
      if (propagateMC) {
        outputSpecs.emplace_back(OutputSpec{ { "outputMC" }, config.mcoutput.origin, config.mcoutput.description, n, Lifetime::Timeframe });
      }
    }
    return std::move(outputSpecs);
  };

  auto& dtb = config.databranch;
  auto& mcb = config.mcbranch;
  return DataProcessorSpec{ config.processName.c_str(),
                            Inputs{}, // no inputs
                            { createOutputSpecs() },
                            AlgorithmSpec(initFunction),
                            Options{
                              { "infile", VariantType::String, "", { "Name of the input file" } },
                              { "treename", VariantType::String, config.defaultTreeName.c_str(), { "Name of input tree" } },
                              { dtb.option.c_str(), VariantType::String, dtb.defval.c_str(), { dtb.help.c_str() } },
                              { mcb.option.c_str(), VariantType::String, mcb.defval.c_str(), { mcb.help.c_str() } },
                              { "nevents", VariantType::Int, -1, { "number of events to run" } },
                              { "terminate-on-eod", VariantType::Bool, true, { "terminate on end-of-data" } },
                            } };
}
} // end namespace TPC
} // end namespace o2
